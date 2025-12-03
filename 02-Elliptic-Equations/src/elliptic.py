"""
Herramientas para resolver problemas elípticos 2D (Poisson/Laplace) en dominios rectangulares.

Funciones principales:
- `build_poisson_2d`: construye la matriz dispersa A y el vector b para el problema discretizado en diferencias finitas.
- `solve_direct`: envoltorio para `spsolve`.
- `solve_cg`: conjugate gradient con opcional precondicionador diagonal o ILU si está disponible.
- `jacobi` / `sor`: solvers iterativos vectorizados para demostraciones y benchmarking.
- `line_relaxation` / `line_sor`: advanced line-based iterative solvers using tridiagonal solves.
- `adi_solve`: Alternating Direction Implicit method with Thomas algorithm.
- `batched_thomas`: wrapper for batched tridiagonal solves with optional Numba/CuPy acceleration.

Convenciones:
- El ordenamiento de unknowns es por filas (x cambia más rápido) o columnas? Aquí usamos el orden por filas: index = i*ny + j (i para x, j para y)
  (esto hace la construcción con Kronecker estándar y coincide con `kron(Ix, Ay) + kron(Ax, Iy)`).
"""

from typing import Tuple, Dict, Optional, List
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Import utilities from chapter 01
# Try to find the repo root and add chapter 01 to path
try:
    # When run as module from repo root
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / '01-Linear-Systems' / 'src'))
except:
    # Fallback for other contexts
    sys.path.insert(0, '01-Linear-Systems/src')

try:
    from linear_systems import tridiagonal_solve, build_discrete_laplacian_1d, compute_residual, compute_relative_error
    CHAPTER01_AVAILABLE = True
except ImportError:
    CHAPTER01_AVAILABLE = False
    print('Warning: Chapter 01 utilities not available. Some advanced solvers disabled.')

# Optional: import parallel solvers from chapter 01
try:
    from parallel_solvers import thomas_solve_numba, thomas_solve_gpu, NUMBA_AVAILABLE, CUPY_AVAILABLE
    PARALLEL_SOLVERS_AVAILABLE = True
except ImportError:
    PARALLEL_SOLVERS_AVAILABLE = False
    NUMBA_AVAILABLE = False
    CUPY_AVAILABLE = False


def _laplacian_1d(n: int, h: float) -> sp.csr_matrix:
    """1D second-difference operator (Dirichlet interior) / h spacing.

    Returns n-by-n sparse matrix for interior points.
    """
    main = -2.0 * np.ones(n) / h**2
    off = 1.0 * np.ones(n - 1) / h**2
    A = sp.diags([main, off, off], [0, -1, 1], format='csr')
    return A


def build_poisson_2d(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                     bc: Optional[Dict[str, Tuple[str, float]]] = None) -> Tuple[sp.csr_matrix, np.ndarray, Tuple[float,float,float,float]]:
    """
    Construye sistema discreto A x = b para -Δ u = f (aquí f=0 típicamente)

    Parameters
    ----------
    nx, ny : int
        Número de nodos en x e y (incluyendo fronteras).
    lx, ly : float
        Longitudes del dominio en x e y.
    bc : dict or str, optional
        Condiciones de frontera. Puede ser:
        - dict: keys 'left','right','bottom','top' con valor ('dirichlet'|'neumann', value)
        - str: 'dirichlet' o 'neumann' aplicado a todos los bordes con valor 0.0
        - None: por defecto Dirichlet 0 en todos los bordes

    Returns
    -------
    A : csr_matrix
        Matriz dispersa del operador sobre incógnitas interiores.
    b : ndarray
        Vector RHS que incorpora condiciones de contorno.
    shape : tuple
        (nx, ny) números de nodos totales (incluyendo fronteras)
    """
    # Handle bc parameter: convert string to dict if needed
    if bc is None:
        bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
              'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 0.0)}
    elif isinstance(bc, str):
        # Convert string 'dirichlet' or 'neumann' to dict for all boundaries
        bc = {'left': (bc, 0.0), 'right': (bc, 0.0),
              'bottom': (bc, 0.0), 'top': (bc, 0.0)}

    hx = lx / (nx - 1)
    hy = ly / (ny - 1)

    # número de incógnitas interiores
    nx_i = nx - 2
    ny_i = ny - 2
    N = nx_i * ny_i

    Ax = _laplacian_1d(nx_i, hx)
    Ay = _laplacian_1d(ny_i, hy)

    # Kronecker sum: A = kron(Ix, Ay) + kron(Ax, Iy)
    I_x = sp.eye(nx_i, format='csr')
    I_y = sp.eye(ny_i, format='csr')

    A = sp.kron(I_x, Ay, format='csr') + sp.kron(Ax, I_y, format='csr')

    # RHS (f=0) plus boundary contributions
    b = np.zeros(N, dtype=float)

    # helper: map interior (i,j) (i=0..nx_i-1, j=0..ny_i-1) to index
    def idx(i, j):
        return i * ny_i + j

    # incorporate Dirichlet boundaries: for each interior node next to boundary, add contribution
    # left boundary (i=0 neighbors)
    bc_left_type, bc_left_val = bc.get('left', ('dirichlet', 0.0))
    bc_right_type, bc_right_val = bc.get('right', ('dirichlet', 0.0))
    bc_bottom_type, bc_bottom_val = bc.get('bottom', ('dirichlet', 0.0))
    bc_top_type, bc_top_val = bc.get('top', ('dirichlet', 0.0))

    # Left boundary
    if nx_i > 0:
        i = 0
        for j in range(ny_i):
            k = idx(i, j)
            if bc_left_type == 'dirichlet':
                b[k] -= bc_left_val / hx**2
            elif bc_left_type == 'neumann':
                # Neumann ∂u/∂n = g approximated by (u_0 - u_ghost)/hx = g
                # leads to contribution of -g/hx in discrete eqn: implement as modification of b
                b[k] -= (-bc_left_val) / hx

    # Right boundary
    if nx_i > 0:
        i = nx_i - 1
        for j in range(ny_i):
            k = idx(i, j)
            if bc_right_type == 'dirichlet':
                b[k] -= bc_right_val / hx**2
            elif bc_right_type == 'neumann':
                b[k] -= (bc_right_val) / hx

    # Bottom boundary (j=0)
    if ny_i > 0:
        j = 0
        for i in range(nx_i):
            k = idx(i, j)
            if bc_bottom_type == 'dirichlet':
                b[k] -= bc_bottom_val / hy**2
            elif bc_bottom_type == 'neumann':
                b[k] -= (-bc_bottom_val) / hy

    # Top boundary (j=ny_i-1)
    if ny_i > 0:
        j = ny_i - 1
        for i in range(nx_i):
            k = idx(i, j)
            if bc_top_type == 'dirichlet':
                b[k] -= bc_top_val / hy**2
            elif bc_top_type == 'neumann':
                b[k] -= (bc_top_val) / hy

    return A.tocsr(), b, (nx, ny, hx, hy)


def build_variable_coeff_2d(nx: int, ny: int, kappa: np.ndarray,
                            source: Optional[np.ndarray] = None,
                            lx: float = 1.0, ly: float = 1.0,
                            bc: Optional[Dict[str, Tuple[str, float]]] = None) -> Tuple[sp.csr_matrix, np.ndarray, Dict]:
    """
    Construye sistema discreto para ecuación con coeficientes variables:
    -∇·(κ(x,y)∇u) = f(x,y)
    
    Usa diferencias finitas con promediado armónico de κ en las caras de las celdas.
    
    Parameters
    ----------
    nx, ny : int
        Número de nodos en x e y (incluyendo fronteras)
    kappa : (ny, nx) array
        Coeficiente de difusividad/conductividad en cada nodo de la grilla
    source : (ny, nx) array or None
        Término fuente f(x,y). Si None, usa f=0
    lx, ly : float
        Longitudes del dominio
    bc : dict or str, optional
        Condiciones de frontera (mismo formato que build_poisson_2d)
        
    Returns
    -------
    A : csr_matrix
        Matriz dispersa del operador
    b : ndarray
        Vector RHS incluyendo fuente y condiciones de contorno
    metadata : dict
        Información adicional (nx, ny, hx, hy, kappa_min, kappa_max)
        
    Notes
    -----
    Para -∇·(κ∇u), la discretización en el punto (i,j) es:
    
    -[(κ_{i+1/2,j}(u_{i+1,j} - u_{i,j}) - κ_{i-1/2,j}(u_{i,j} - u_{i-1,j}))/hx²
     +(κ_{i,j+1/2}(u_{i,j+1} - u_{i,j}) - κ_{i,j-1/2}(u_{i,j} - u_{i,j-1}))/hy²]
    
    Usamos promediado armónico para κ en las caras: κ_{i+1/2,j} = 2/(1/κ_i + 1/κ_{i+1})
    
    Examples
    --------
    >>> # Material con dos capas: kappa=1 en mitad izquierda, kappa=10 en mitad derecha
    >>> nx, ny = 51, 51
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> X, Y = np.meshgrid(x, y)
    >>> kappa = np.where(X < 0.5, 1.0, 10.0)
    >>> A, b, meta = build_variable_coeff_2d(nx, ny, kappa)
    """
    # Handle bc parameter
    if bc is None:
        bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
              'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 0.0)}
    elif isinstance(bc, str):
        bc = {'left': (bc, 0.0), 'right': (bc, 0.0),
              'bottom': (bc, 0.0), 'top': (bc, 0.0)}
    
    # Validate kappa shape
    if kappa.shape != (ny, nx):
        raise ValueError(f"kappa must have shape ({ny}, {nx}), got {kappa.shape}")
    
    if source is not None and source.shape != (ny, nx):
        raise ValueError(f"source must have shape ({ny}, {nx}), got {source.shape}")
    
    hx = lx / (nx - 1)
    hy = ly / (ny - 1)
    
    # Interior points
    nx_i = nx - 2
    ny_i = ny - 2
    N = nx_i * ny_i
    
    # Index mapping: (i,j) interior -> global index
    def idx(i, j):
        return i * ny_i + j
    
    # Build sparse matrix using COO format (easier to construct)
    row_indices = []
    col_indices = []
    values = []
    
    # RHS vector
    b = np.zeros(N)
    
    # Add source term if provided
    if source is not None:
        b = source[1:-1, 1:-1].flatten()  # Interior points only
    
    # Harmonic mean for interface diffusivity
    def harmonic_mean(k1, k2):
        """Harmonic mean: 2/(1/k1 + 1/k2) = 2*k1*k2/(k1+k2)"""
        if k1 <= 0 or k2 <= 0:
            return 0.0
        return 2 * k1 * k2 / (k1 + k2)
    
    # Build stencil for each interior point
    for i in range(nx_i):
        for j in range(ny_i):
            k = idx(i, j)
            
            # Current point in global grid (including boundaries)
            gi = i + 1
            gj = j + 1
            
            # Center coefficient (will accumulate contributions)
            center_coeff = 0.0
            
            # --- X-direction: left and right neighbors ---
            
            # Right neighbor (i+1, j)
            if i < nx_i - 1:
                # Interior-interior connection
                kappa_right = harmonic_mean(kappa[gj, gi], kappa[gj, gi+1])
                coeff = kappa_right / hx**2
                row_indices.append(k)
                col_indices.append(idx(i+1, j))
                values.append(coeff)
                center_coeff -= coeff
            else:
                # Right boundary
                bc_type, bc_val = bc.get('right', ('dirichlet', 0.0))
                if bc_type == 'dirichlet':
                    kappa_right = harmonic_mean(kappa[gj, gi], kappa[gj, gi+1])
                    coeff = kappa_right / hx**2
                    center_coeff -= coeff
                    b[k] -= coeff * bc_val
                elif bc_type == 'neumann':
                    # Neumann BC: flux = -kappa * du/dn = g
                    # Discretization: -kappa * (u_boundary - u_interior)/hx = g
                    # No modification to matrix, add to RHS
                    b[k] -= bc_val * kappa[gj, gi] / hx
            
            # Left neighbor (i-1, j)
            if i > 0:
                # Interior-interior connection
                kappa_left = harmonic_mean(kappa[gj, gi], kappa[gj, gi-1])
                coeff = kappa_left / hx**2
                row_indices.append(k)
                col_indices.append(idx(i-1, j))
                values.append(coeff)
                center_coeff -= coeff
            else:
                # Left boundary
                bc_type, bc_val = bc.get('left', ('dirichlet', 0.0))
                if bc_type == 'dirichlet':
                    kappa_left = harmonic_mean(kappa[gj, gi], kappa[gj, gi-1])
                    coeff = kappa_left / hx**2
                    center_coeff -= coeff
                    b[k] -= coeff * bc_val
                elif bc_type == 'neumann':
                    b[k] += bc_val * kappa[gj, gi] / hx
            
            # --- Y-direction: bottom and top neighbors ---
            
            # Top neighbor (i, j+1)
            if j < ny_i - 1:
                # Interior-interior connection
                kappa_top = harmonic_mean(kappa[gj, gi], kappa[gj+1, gi])
                coeff = kappa_top / hy**2
                row_indices.append(k)
                col_indices.append(idx(i, j+1))
                values.append(coeff)
                center_coeff -= coeff
            else:
                # Top boundary
                bc_type, bc_val = bc.get('top', ('dirichlet', 0.0))
                if bc_type == 'dirichlet':
                    kappa_top = harmonic_mean(kappa[gj, gi], kappa[gj+1, gi])
                    coeff = kappa_top / hy**2
                    center_coeff -= coeff
                    b[k] -= coeff * bc_val
                elif bc_type == 'neumann':
                    b[k] -= bc_val * kappa[gj, gi] / hy
            
            # Bottom neighbor (i, j-1)
            if j > 0:
                # Interior-interior connection
                kappa_bottom = harmonic_mean(kappa[gj, gi], kappa[gj-1, gi])
                coeff = kappa_bottom / hy**2
                row_indices.append(k)
                col_indices.append(idx(i, j-1))
                values.append(coeff)
                center_coeff -= coeff
            else:
                # Bottom boundary
                bc_type, bc_val = bc.get('bottom', ('dirichlet', 0.0))
                if bc_type == 'dirichlet':
                    kappa_bottom = harmonic_mean(kappa[gj, gi], kappa[gj-1, gi])
                    coeff = kappa_bottom / hy**2
                    center_coeff -= coeff
                    b[k] -= coeff * bc_val
                elif bc_type == 'neumann':
                    b[k] += bc_val * kappa[gj, gi] / hy
            
            # Add center coefficient
            row_indices.append(k)
            col_indices.append(k)
            values.append(center_coeff)
    
    # Construct sparse matrix
    A = sp.coo_matrix((values, (row_indices, col_indices)), shape=(N, N))
    A = A.tocsr()
    
    metadata = {
        'nx': nx,
        'ny': ny,
        'hx': hx,
        'hy': hy,
        'kappa_min': np.min(kappa),
        'kappa_max': np.max(kappa),
        'kappa_ratio': np.max(kappa) / np.min(kappa)
    }
    
    return A, b, metadata


def solve_direct(A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
    """Resuelve A x = b usando spsolve (directo).
    """
    return spla.spsolve(A, b)


def solve_cg(A: sp.spmatrix, b: np.ndarray, tol: float = 1e-8, maxiter: Optional[int] = None,
             precondition: Optional[str] = 'jacobi') -> Tuple[np.ndarray, Dict]:
    """Resolve con conjugate gradient (para A simétrica negativa-definida aquí).

    precondition: 'jacobi' para diagonally scaled preconditioner, 'ilu' para ILU (si está disponible), or None.
    Returns solution and info dict with convergence info.
    """
    M = None
    if precondition == 'jacobi':
        diag = A.diagonal()
        # avoid zeros
        diag[diag == 0] = 1.0
        M = spla.LinearOperator(A.shape, lambda x: x / diag)
    elif precondition == 'ilu':
        try:
            ilu = spla.spilu(A.tocsc())
            M = spla.LinearOperator(A.shape, ilu.solve)
        except Exception:
            M = None

    x, info = spla.cg(A, b, atol=tol, maxiter=maxiter, M=M)
    return x, {'info': info}


def jacobi(A: sp.spmatrix, b: np.ndarray, x0: Optional[np.ndarray] = None, tol: float = 1e-8, maxiter: int = 10000) -> Tuple[np.ndarray, int, float]:
    """Jacobi iterativo (vectorizado). Retorna (x, iterations, residual).
    """
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    D = A.diagonal()
    R = A - sp.diags(D, 0)

    for k in range(maxiter):
        x_new = (b - R.dot(x)) / D
        r = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if r < tol:
            return x, k + 1, r
    return x, maxiter, r


def sor(A: sp.spmatrix, b: np.ndarray, omega: float = 1.5, x0: Optional[np.ndarray] = None,
        tol: float = 1e-8, maxiter: int = 10000) -> Tuple[np.ndarray, int, float]:
    """Successive Over-Relaxation (simple sparse implementation using CSR data).
    """
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    A_csr = A.tocsr()
    D = A_csr.diagonal()

    for k in range(maxiter):
        x_old = x.copy()
        # loop over rows (could be vectorized but simple implementation is clearer)
        for i in range(n):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i+1]
            cols = A_csr.indices[row_start:row_end]
            vals = A_csr.data[row_start:row_end]
            Ai = 0.0
            for cj, v in zip(cols, vals):
                if cj != i:
                    Ai += v * x[cj]
            x[i] = (1 - omega) * x[i] + (omega / D[i]) * (b[i] - Ai)

        r = np.linalg.norm(x - x_old, ord=np.inf)
        if r < tol:
            return x, k + 1, r
    return x, maxiter, r


def batched_thomas(d_list: List[np.ndarray], u_list: List[np.ndarray], 
                   o_list: List[np.ndarray], b_list: List[np.ndarray],
                   backend: str = 'numpy') -> List[np.ndarray]:
    """Batched tridiagonal solve using chapter 01 utilities.
    
    Parameters
    ----------
    d_list, u_list, o_list, b_list : list of arrays
        Lists of diagonal arrays and RHS vectors for multiple systems.
    backend : str
        'numpy' (default), 'numba' (if available), or 'gpu' (if CuPy available).
    
    Returns
    -------
    list of np.ndarray
        Solutions for each system.
    """
    if not CHAPTER01_AVAILABLE:
        raise RuntimeError('Chapter 01 utilities required for batched_thomas')
    
    solutions = []
    if backend == 'numba' and NUMBA_AVAILABLE and PARALLEL_SOLVERS_AVAILABLE:
        for d, u, o, b in zip(d_list, u_list, o_list, b_list):
            x = thomas_solve_numba(d, u, o, b)
            solutions.append(x)
    elif backend == 'gpu' and CUPY_AVAILABLE and PARALLEL_SOLVERS_AVAILABLE:
        for d, u, o, b in zip(d_list, u_list, o_list, b_list):
            x = thomas_solve_gpu(d, u, o, b)
            solutions.append(x)
    else:
        # fallback to numpy
        for d, u, o, b in zip(d_list, u_list, o_list, b_list):
            x = tridiagonal_solve(d, u, o, b, modify_inplace=False)
            solutions.append(x)
    return solutions


def line_relaxation(nx: int, ny: int, hx: float, hy: float, bc: Dict, 
                    b_vec: np.ndarray, x0: Optional[np.ndarray] = None,
                    tol: float = 1e-8, maxiter: int = 1000, 
                    omega: float = 1.0, axis: str = 'x') -> Tuple[np.ndarray, int, float]:
    """Line-relaxation solver: solves tridiagonal systems along lines.
    
    Parameters
    ----------
    nx, ny : int
        Number of interior points in x and y.
    hx, hy : float
        Grid spacing.
    bc : dict
        Boundary conditions.
    b_vec : np.ndarray
        RHS vector (flattened, length nx*ny).
    x0 : np.ndarray, optional
        Initial guess.
    tol : float
        Convergence tolerance (inf-norm of change).
    maxiter : int
        Maximum iterations.
    omega : float
        Relaxation parameter (1.0 = Gauss-Seidel, >1 = SOR).
    axis : str
        'x' or 'y' — which direction to solve tridiagonal systems.
    
    Returns
    -------
    x : np.ndarray
        Solution vector.
    iters : int
        Number of iterations.
    residual : float
        Final change norm.
    """
    if not CHAPTER01_AVAILABLE:
        raise RuntimeError('Chapter 01 utilities required for line_relaxation')
    
    x = np.zeros(nx * ny) if x0 is None else x0.copy()
    U = x.reshape((nx, ny), order='C')  # shape: (nx, ny) with ordering i*ny+j
    b_grid = b_vec.reshape((nx, ny), order='C')
    
    # Build tridiagonal operators for x and y directions (these include 1/h^2 scaling)
    d_x, u_x, o_x = build_discrete_laplacian_1d(nx, hx)
    d_y, u_y, o_y = build_discrete_laplacian_1d(ny, hy)
    
    for it in range(maxiter):
        U_old = U.copy()
        
        if axis == 'x':
            # For each horizontal line j, solve in x-direction
            # The full 2D operator has diagonal: -2/hx² - 2/hy²
            # We need to solve: (A_2D) u[:, j] = b[:, j]
            # Using: A_x u[:, j] + contributions from y-neighbors = b[:, j]
            for j in range(ny):
                rhs = b_grid[:, j].copy()
                # Add contributions from y-neighbors (these come from A_y off-diagonals)
                if j > 0:
                    rhs -= U[:, j-1] / hy**2
                if j < ny - 1:
                    rhs -= U[:, j+1] / hy**2
                # Now we need to add the -2/hy² term to the diagonal of A_x
                # A_x has -2/hx², we need -2/hx² - 2/hy²
                d_modified = d_x - 2.0 / hy**2
                # Solve with modified diagonal
                x_line = tridiagonal_solve(d_modified, u_x, o_x, rhs, modify_inplace=False)
                U[:, j] = omega * x_line + (1 - omega) * U[:, j]
        else:  # axis == 'y'
            # For each vertical line i, solve in y-direction  
            for i in range(nx):
                rhs = b_grid[i, :].copy()
                # Add contributions from x-neighbors
                if i > 0:
                    rhs -= U[i-1, :] / hx**2
                if i < nx - 1:
                    rhs -= U[i+1, :] / hx**2
                # Modify diagonal to include -2/hx² term
                d_modified = d_y - 2.0 / hx**2
                # Solve with modified diagonal
                y_line = tridiagonal_solve(d_modified, u_y, o_y, rhs, modify_inplace=False)
                U[i, :] = omega * y_line + (1 - omega) * U[i, :]
        
        change = np.linalg.norm(U - U_old, ord=np.inf)
        if change < tol:
            return U.flatten(order='C'), it + 1, change
    
    return U.flatten(order='C'), maxiter, change


def adi_solve(nx: int, ny: int, hx: float, hy: float, bc: Dict,
              b_vec: np.ndarray, x0: Optional[np.ndarray] = None,
              tol: float = 1e-8, maxiter: int = 500) -> Tuple[np.ndarray, int, float]:
    """Alternating Direction Implicit (ADI) method using Thomas algorithm.
    
    Parameters
    ----------
    nx, ny : int
        Interior grid points (NOT including boundaries).
    hx, hy : float
        Grid spacing.
    bc : dict
        Boundary conditions.
    b_vec : np.ndarray
        RHS vector (flattened, length nx*ny for interior points).
    x0 : np.ndarray, optional
        Initial guess.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.
    
    Returns
    -------
    x : np.ndarray
        Solution vector (flattened, length nx*ny).
    iters : int
        Iterations taken.
    residual : float
        Final residual norm.
    """
    if not CHAPTER01_AVAILABLE:
        raise RuntimeError('Chapter 01 utilities required for ADI')
    
    # b_vec has shape (nx*ny) where nx, ny are INTERIOR points
    # Verify dimensions
    expected_size = nx * ny
    if b_vec.size != expected_size:
        raise ValueError(f"b_vec has size {b_vec.size}, expected {expected_size} (nx={nx}, ny={ny})")
    
    x = np.zeros(nx * ny) if x0 is None else x0.copy()
    U = x.reshape((nx, ny), order='C')
    b_grid = b_vec.reshape((nx, ny), order='C')
    
    # ADI for steady-state: Line-Gauss-Seidel alternating between x and y sweeps
    # U[i,j] where i=0..nx-1 (x-direction), j=0..ny-1 (y-direction)
    # Indexing: idx(i,j) = i*ny + j (row-major, i is row/x-index, j is column/y-index)
    
    d_x, u_x, o_x = build_discrete_laplacian_1d(nx, hx)
    d_y, u_y, o_y = build_discrete_laplacian_1d(ny, hy)
    
    for it in range(maxiter):
        U_old = U.copy()
        
        # X-sweep: For each horizontal line (constant j), solve implicitly along i (x-direction)
        # The full 2D Laplacian has diagonal -2/hx² - 2/hy²
        for j in range(ny):
            rhs = b_grid[:, j].copy()
            # Move y-neighbors to RHS
            if j > 0:
                rhs -= U[:, j-1] / hy**2
            if j < ny - 1:
                rhs -= U[:, j+1] / hy**2
            # Modify diagonal to account for 2D operator
            d_modified = d_x - 2.0 / hy**2
            # Solve tridiagonal system along x
            U[:, j] = tridiagonal_solve(d_modified, u_x, o_x, rhs, modify_inplace=False)
        
        # Y-sweep: For each vertical line (constant i), solve implicitly along j (y-direction)
        for i in range(nx):
            rhs = b_grid[i, :].copy()
            # Move x-neighbors to RHS (using updated values from x-sweep)
            if i > 0:
                rhs -= U[i-1, :] / hx**2
            if i < nx - 1:
                rhs -= U[i+1, :] / hx**2
            # Modify diagonal
            d_modified = d_y - 2.0 / hx**2
            # Solve tridiagonal system along y
            U[i, :] = tridiagonal_solve(d_modified, u_y, o_y, rhs, modify_inplace=False)
        
        change = np.linalg.norm(U - U_old, ord=np.inf)
        if change < tol:
            return U.flatten(order='C'), it + 1, change
    
    return U.flatten(order='C'), maxiter, change


if __name__ == '__main__':
    # pequeño demo cuando se ejecuta directamente
    import matplotlib.pyplot as plt

    nx, ny = 50, 50
    bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
          'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 1.0)}

    A, b, meta = build_poisson_2d(nx, ny, lx=1.0, ly=1.0, bc=bc)
    print('A shape', A.shape)
    x = solve_direct(A, b)

    nx_tot, ny_tot, hx, hy = meta
    nx_i = nx_tot - 2
    ny_i = ny_tot - 2
    U = np.zeros((ny_tot, nx_tot))
    # interior mapping: i index in x from 0..nx_i-1, j in y
    k = 0
    for i in range(nx_i):
        for j in range(ny_i):
            U[j+1, i+1] = x[k]
            k += 1

    # set boundaries
    U[0, :] = bc['bottom'][1]
    U[-1, :] = bc['top'][1]
    U[:, 0] = bc['left'][1]
    U[:, -1] = bc['right'][1]

    xg = np.linspace(0, 1, nx_tot)
    yg = np.linspace(0, 1, ny_tot)
    X, Y = np.meshgrid(xg, yg)

    plt.figure()
    plt.contourf(X, Y, U, 50, cmap='coolwarm')
    plt.colorbar()
    plt.title('Solución Laplace - demo')
    plt.show()


# ============================================================================
# MULTIGRID METHODS (Geometric Multigrid - GMG)
# ============================================================================

def restrict_injection(u_fine: np.ndarray) -> np.ndarray:
    """
    Injection restriction operator: directly sample every other point.
    
    Transfer from fine grid (2h) to coarse grid (h) by injection (simple sampling).
    This is the simplest restriction but can be noisy.
    
    Parameters
    ----------
    u_fine : (ny_fine, nx_fine) array
        Solution on fine grid
        
    Returns
    -------
    u_coarse : (ny_coarse, nx_coarse) array
        Restricted solution on coarse grid
        
    Notes
    -----
    Assumes ny_fine, nx_fine are odd (so (ny_fine-1)/2, (nx_fine-1)/2 are integers).
    """
    return u_fine[::2, ::2]


def restrict_full_weighting(u_fine: np.ndarray) -> np.ndarray:
    """
    Full-weighting restriction operator (recommended for Poisson).
    
    Uses 9-point weighted stencil:
        [1  2  1]
        [2  4  2] / 16
        [1  2  1]
    
    This provides better smoothing of high-frequency errors than injection.
    
    Parameters
    ----------
    u_fine : (ny_fine, nx_fine) array
        Solution on fine grid
        
    Returns
    -------
    u_coarse : (ny_coarse, nx_coarse) array
        Restricted solution on coarse grid
        
    Notes
    -----
    Standard choice for geometric multigrid. Preserves L2 norm better than injection.
    """
    ny_fine, nx_fine = u_fine.shape
    ny_coarse = (ny_fine - 1) // 2 + 1
    nx_coarse = (nx_fine - 1) // 2 + 1
    
    u_coarse = np.zeros((ny_coarse, nx_coarse))
    
    for i in range(ny_coarse):
        for j in range(nx_coarse):
            i_f = 2 * i
            j_f = 2 * j
            
            # Center point weight 4
            u_coarse[i, j] = 4.0 * u_fine[i_f, j_f]
            
            # Edge neighbors weight 2
            if i_f > 0:
                u_coarse[i, j] += 2.0 * u_fine[i_f - 1, j_f]
            if i_f < ny_fine - 1:
                u_coarse[i, j] += 2.0 * u_fine[i_f + 1, j_f]
            if j_f > 0:
                u_coarse[i, j] += 2.0 * u_fine[i_f, j_f - 1]
            if j_f < nx_fine - 1:
                u_coarse[i, j] += 2.0 * u_fine[i_f, j_f + 1]
            
            # Corner neighbors weight 1
            if i_f > 0 and j_f > 0:
                u_coarse[i, j] += 1.0 * u_fine[i_f - 1, j_f - 1]
            if i_f > 0 and j_f < nx_fine - 1:
                u_coarse[i, j] += 1.0 * u_fine[i_f - 1, j_f + 1]
            if i_f < ny_fine - 1 and j_f > 0:
                u_coarse[i, j] += 1.0 * u_fine[i_f + 1, j_f - 1]
            if i_f < ny_fine - 1 and j_f < nx_fine - 1:
                u_coarse[i, j] += 1.0 * u_fine[i_f + 1, j_f + 1]
            
            u_coarse[i, j] /= 16.0
    
    return u_coarse


def prolong_linear(u_coarse: np.ndarray, ny_fine: int, nx_fine: int) -> np.ndarray:
    """
    Bilinear interpolation prolongation operator.
    
    Transfer from coarse grid to fine grid using bilinear interpolation.
    This is the standard prolongation for geometric multigrid.
    
    Parameters
    ----------
    u_coarse : (ny_coarse, nx_coarse) array
        Solution on coarse grid
    ny_fine, nx_fine : int
        Target fine grid dimensions
        
    Returns
    -------
    u_fine : (ny_fine, nx_fine) array
        Prolonged solution on fine grid
        
    Notes
    -----
    Uses stencil:
        [1  2  1]
        [2  4  2] / 4  (applied appropriately for interpolation)
    """
    ny_coarse, nx_coarse = u_coarse.shape
    u_fine = np.zeros((ny_fine, nx_fine))
    
    # Direct injection at coarse points
    for i in range(ny_coarse):
        for j in range(nx_coarse):
            u_fine[2*i, 2*j] = u_coarse[i, j]
    
    # Interpolate at edge midpoints
    for i in range(ny_coarse):
        for j in range(nx_coarse - 1):
            u_fine[2*i, 2*j + 1] = 0.5 * (u_coarse[i, j] + u_coarse[i, j + 1])
    
    for i in range(ny_coarse - 1):
        for j in range(nx_coarse):
            u_fine[2*i + 1, 2*j] = 0.5 * (u_coarse[i, j] + u_coarse[i + 1, j])
    
    # Interpolate at cell centers
    for i in range(ny_coarse - 1):
        for j in range(nx_coarse - 1):
            u_fine[2*i + 1, 2*j + 1] = 0.25 * (u_coarse[i, j] + u_coarse[i, j + 1] + 
                                                u_coarse[i + 1, j] + u_coarse[i + 1, j + 1])
    
    return u_fine


def residual_2d(u: np.ndarray, f: np.ndarray, hx: float, hy: float) -> np.ndarray:
    """
    Compute residual r = f - A*u for 2D Poisson equation.
    
    Uses 5-point stencil for discrete Laplacian.
    
    Parameters
    ----------
    u : (ny, nx) array
        Current solution
    f : (ny, nx) array
        Right-hand side
    hx, hy : float
        Grid spacing in x and y directions
        
    Returns
    -------
    r : (ny, nx) array
        Residual r = f - A*u
        
    Notes
    -----
    Interior points use standard 5-point stencil.
    Boundary points are assumed fixed (Dirichlet), so residual is zero there.
    """
    ny, nx = u.shape
    r = np.zeros_like(u)
    
    hx2 = hx * hx
    hy2 = hy * hy
    
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            laplacian_u = ((u[i, j-1] - 2*u[i, j] + u[i, j+1]) / hx2 +
                          (u[i-1, j] - 2*u[i, j] + u[i+1, j]) / hy2)
            r[i, j] = f[i, j] - (-laplacian_u)  # Note: -∇²u = f
    
    return r


def smooth_gauss_seidel_rb(u: np.ndarray, f: np.ndarray, hx: float, hy: float, 
                           iterations: int = 1) -> np.ndarray:
    """
    Red-Black Gauss-Seidel smoother for 2D Poisson equation.
    
    Red-black ordering allows parallel updates within each color.
    Excellent for removing high-frequency errors (smoothing property).
    
    Parameters
    ----------
    u : (ny, nx) array
        Current solution (modified in place)
    f : (ny, nx) array
        Right-hand side
    hx, hy : float
        Grid spacing
    iterations : int
        Number of smoothing sweeps
        
    Returns
    -------
    u : (ny, nx) array
        Smoothed solution
        
    Notes
    -----
    Red points: (i+j) even, Black points: (i+j) odd.
    This ordering allows vectorization and parallelization.
    """
    ny, nx = u.shape
    hx2 = hx * hx
    hy2 = hy * hy
    
    # Diagonal coefficient for 5-point stencil
    diag = -2.0 / hx2 - 2.0 / hy2
    
    for _ in range(iterations):
        # Red points: (i+j) even
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if (i + j) % 2 == 0:
                    rhs = f[i, j]
                    neighbors = ((u[i, j-1] + u[i, j+1]) / hx2 +
                               (u[i-1, j] + u[i+1, j]) / hy2)
                    u[i, j] = (rhs + neighbors) / (-diag)
        
        # Black points: (i+j) odd
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if (i + j) % 2 == 1:
                    rhs = f[i, j]
                    neighbors = ((u[i, j-1] + u[i, j+1]) / hx2 +
                               (u[i-1, j] + u[i+1, j]) / hy2)
                    u[i, j] = (rhs + neighbors) / (-diag)
    
    return u


def v_cycle(u: np.ndarray, f: np.ndarray, hx: float, hy: float, 
           levels: int = 3, pre_smooth: int = 2, post_smooth: int = 2) -> np.ndarray:
    """
    V-cycle multigrid iteration for 2D Poisson equation.
    
    Classical V-cycle: smooth -> restrict -> recurse -> prolong -> smooth
    
    Parameters
    ----------
    u : (ny, nx) array
        Initial guess (modified in place)
    f : (ny, nx) array
        Right-hand side
    hx, hy : float
        Grid spacing on current level
    levels : int
        Number of grid levels (1 = no multigrid, just smoothing)
    pre_smooth : int
        Smoothing iterations before coarse-grid correction
    post_smooth : int
        Smoothing iterations after coarse-grid correction
        
    Returns
    -------
    u : (ny, nx) array
        Improved solution after V-cycle
        
    Notes
    -----
    V-cycle complexity: O(n) per iteration where n = total grid points.
    Typically reduces error by factor 0.1-0.3 per cycle.
    On coarsest grid, applies heavy smoothing (50 iterations) instead of direct solve.
    """
    ny, nx = u.shape
    
    # Base case: coarsest grid or single level
    if levels == 1 or ny <= 5 or nx <= 5:
        # On coarsest grid, just smooth heavily to converge
        # (Direct solve would require building matrix which is complex for arbitrary RHS)
        u = smooth_gauss_seidel_rb(u, f, hx, hy, iterations=50)
        return u
    
    # Pre-smoothing
    u = smooth_gauss_seidel_rb(u, f, hx, hy, iterations=pre_smooth)
    
    # Compute residual
    r = residual_2d(u, f, hx, hy)
    
    # Restrict residual to coarse grid
    r_coarse = restrict_full_weighting(r)
    
    # Coarse grid correction: solve A_coarse * e_coarse = r_coarse
    ny_coarse, nx_coarse = r_coarse.shape
    e_coarse = np.zeros_like(r_coarse)
    hx_coarse = 2.0 * hx
    hy_coarse = 2.0 * hy
    
    # Recursive V-cycle on coarse grid
    e_coarse = v_cycle(e_coarse, r_coarse, hx_coarse, hy_coarse, 
                      levels - 1, pre_smooth, post_smooth)
    
    # Prolong correction to fine grid
    e_fine = prolong_linear(e_coarse, ny, nx)
    
    # Apply correction
    u += e_fine
    
    # Post-smoothing
    u = smooth_gauss_seidel_rb(u, f, hx, hy, iterations=post_smooth)
    
    return u


def multigrid_solve(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                   bc: Dict = None, source_term: Optional[np.ndarray] = None,
                   tol: float = 1e-8, maxiter: int = 50, levels: int = None,
                   verbose: bool = False) -> Tuple[np.ndarray, int, List[float]]:
    """
    Geometric Multigrid solver for 2D Poisson equation.
    
    Solves -∇²u = f on rectangular domain with Dirichlet boundary conditions
    using V-cycle multigrid method.
    
    Parameters
    ----------
    nx, ny : int
        Number of interior grid points in x and y
    lx, ly : float
        Domain size in x and y
    bc : dict
        Boundary conditions (same format as build_poisson_2d)
    source_term : (ny, nx) array or None
        Source term f. If None, uses f=0
    tol : float
        Convergence tolerance (relative residual)
    maxiter : int
        Maximum number of V-cycles
    levels : int or None
        Number of multigrid levels. If None, computed automatically.
    verbose : bool
        Print convergence information
        
    Returns
    -------
    u : (ny+2, nx+2) array
        Solution including boundary points
    iterations : int
        Number of V-cycles performed
    residual_history : list of float
        Residual norm at each iteration
        
    Examples
    --------
    >>> nx, ny = 127, 127  # Odd numbers work best (power of 2 + 1)
    >>> bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
    ...       'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    >>> u, iters, hist = multigrid_solve(nx, ny, bc=bc, verbose=True)
    >>> print(f"Converged in {iters} V-cycles")
    
    Notes
    -----
    Best performance with grid sizes nx = ny = 2^k - 1 (e.g., 31, 63, 127, 255).
    Typical convergence: 5-10 V-cycles independent of grid size (O(n) complexity).
    """
    if bc is None:
        bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
              'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    
    # Determine number of levels automatically
    if levels is None:
        min_size = min(nx, ny)
        levels = int(np.log2(min_size)) - 1  # Stop at ~4x4 grid
        levels = max(2, min(levels, 8))  # Reasonable range
    
    hx = lx / (nx + 1)
    hy = ly / (ny + 1)
    
    # Full grid including boundaries
    ny_tot = ny + 2
    nx_tot = nx + 2
    
    # Initialize solution and RHS
    u = np.zeros((ny_tot, nx_tot))
    f = np.zeros((ny_tot, nx_tot))
    
    # Apply boundary conditions
    u[0, :] = bc.get('bottom', ('dirichlet', 0))[1]
    u[-1, :] = bc.get('top', ('dirichlet', 0))[1]
    u[:, 0] = bc.get('left', ('dirichlet', 0))[1]
    u[:, -1] = bc.get('right', ('dirichlet', 0))[1]
    
    # Set source term
    if source_term is not None:
        if source_term.shape == (ny, nx):
            f[1:-1, 1:-1] = source_term
        else:
            f[1:-1, 1:-1] = source_term.reshape((ny, nx))
    
    # Initial residual
    r0 = np.linalg.norm(residual_2d(u, f, hx, hy))
    if r0 < 1e-14:
        r0 = 1.0  # Avoid division by zero
    
    residual_history = [r0]
    
    if verbose:
        print(f"Multigrid solver: {nx}×{ny} grid, {levels} levels")
        print(f"Initial residual: {r0:.4e}")
    
    # V-cycle iterations
    for iteration in range(maxiter):
        u = v_cycle(u, f, hx, hy, levels=levels, pre_smooth=2, post_smooth=2)
        
        # Check convergence
        r = np.linalg.norm(residual_2d(u, f, hx, hy))
        residual_history.append(r)
        
        rel_res = r / r0
        
        if verbose and (iteration % 5 == 0 or rel_res < tol):
            print(f"  Iteration {iteration + 1}: residual = {r:.4e} (relative: {rel_res:.4e})")
        
        if rel_res < tol:
            if verbose:
                print(f"Converged in {iteration + 1} V-cycles")
            return u, iteration + 1, residual_history
    
    if verbose:
        print(f"Warning: Maximum iterations reached. Final relative residual: {rel_res:.4e}")
    
    return u, maxiter, residual_history
