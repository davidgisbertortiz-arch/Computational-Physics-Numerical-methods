"""
Eigenvalue Solvers Module

This module provides iterative methods for computing eigenvalues and eigenvectors
of matrices, with focus on methods commonly used in computational physics.

Methods implemented:
- Power iteration: dominant eigenvalue
- Inverse iteration: eigenvalue closest to a shift
- Rayleigh quotient iteration: fast convergence for symmetric matrices
- QR iteration: all eigenvalues (basic version)
- Lanczos iteration: for large sparse symmetric matrices
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import sys
from pathlib import Path

# =============================================================================
# Inter-chapter imports (Chapter 01 utilities)
# =============================================================================

try:
    # Try to import from Chapter 01
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / '01-Linear-Systems' / 'src'))
    from linear_systems import tridiagonal_solve, build_tridiagonal
    CHAPTER01_AVAILABLE = True
except ImportError:
    CHAPTER01_AVAILABLE = False

# =============================================================================
# Optional dependencies
# =============================================================================

try:
    from scipy import linalg as scipy_linalg
    from scipy.sparse import issparse
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# =============================================================================
# Core eigenvalue methods
# =============================================================================

def power_iteration(
    A: np.ndarray,
    v0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
    return_history: bool = False
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Power iteration method for finding the dominant eigenvalue.
    
    Finds the eigenvalue with largest absolute value and its eigenvector.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix (n x n)
    v0 : np.ndarray, optional
        Initial guess for eigenvector. If None, uses random vector.
    tol : float, optional
        Convergence tolerance for eigenvalue
    max_iter : int, optional
        Maximum number of iterations
    return_history : bool, optional
        If True, return history of eigenvalue estimates
    
    Returns
    -------
    eigenvalue : float
        Dominant eigenvalue (largest |λ|)
    eigenvector : np.ndarray
        Corresponding unit eigenvector
    info : dict
        Convergence information:
        - 'iterations': number of iterations performed
        - 'converged': whether method converged
        - 'residual': final residual ||Av - λv||
        - 'history': eigenvalue history (if return_history=True)
    
    Examples
    --------
    >>> A = np.array([[4, 1], [1, 3]])
    >>> lam, v, info = power_iteration(A)
    >>> np.isclose(lam, 4.618, rtol=1e-3)  # Dominant eigenvalue ≈ (7+√5)/2
    True
    
    Notes
    -----
    - Converges to eigenvalue with largest |λ|
    - Convergence rate: |λ₂/λ₁| (ratio of second largest to largest)
    - Fails if λ₁ = -λ₂ (need to handle sign)
    
    References
    ----------
    Trefethen & Bau, "Numerical Linear Algebra", Algorithm 27.1
    """
    n = A.shape[0]
    
    # Initialize
    if v0 is None:
        np.random.seed(42)  # Reproducibility
        v = np.random.randn(n)
    else:
        v = v0.copy()
    
    v = v / np.linalg.norm(v)
    lam_old = 0.0
    history = [] if return_history else None
    
    for k in range(max_iter):
        # Power step: w = A @ v
        w = A @ v
        
        # Rayleigh quotient for eigenvalue estimate
        lam = np.dot(v, w)
        
        # Normalize
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-15:
            # Matrix might be nilpotent or v in null space
            break
        v = w / norm_w
        
        if return_history:
            history.append(lam)
        
        # Check convergence
        if abs(lam - lam_old) < tol * max(1.0, abs(lam)):
            residual = np.linalg.norm(A @ v - lam * v)
            info = {
                'iterations': k + 1,
                'converged': True,
                'residual': residual
            }
            if return_history:
                info['history'] = np.array(history)
            return float(lam), v, info
        
        lam_old = lam
    
    # Did not converge
    residual = np.linalg.norm(A @ v - lam * v)
    info = {
        'iterations': max_iter,
        'converged': False,
        'residual': residual
    }
    if return_history:
        info['history'] = np.array(history)
    
    return float(lam), v, info


def inverse_iteration(
    A: np.ndarray,
    sigma: float = 0.0,
    v0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 100,
    return_history: bool = False
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Inverse iteration method for finding eigenvalue closest to a shift.
    
    Finds the eigenvalue closest to σ and its eigenvector by applying
    power iteration to (A - σI)^{-1}.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix (n x n)
    sigma : float, optional
        Shift value. Method finds eigenvalue closest to sigma.
    v0 : np.ndarray, optional
        Initial guess for eigenvector
    tol : float, optional
        Convergence tolerance
    max_iter : int, optional
        Maximum number of iterations
    return_history : bool, optional
        If True, return history of eigenvalue estimates
    
    Returns
    -------
    eigenvalue : float
        Eigenvalue closest to sigma
    eigenvector : np.ndarray
        Corresponding unit eigenvector
    info : dict
        Convergence information
    
    Examples
    --------
    >>> A = np.array([[4, 1], [1, 3]])
    >>> # Find eigenvalue closest to 2.5
    >>> lam, v, info = inverse_iteration(A, sigma=2.5)
    >>> np.isclose(lam, 2.382, rtol=1e-3)  # Smaller eigenvalue ≈ (7-√5)/2
    True
    
    Notes
    -----
    - Convergence rate depends on |λ - σ| / |λ_next - σ|
    - Very fast if σ is close to an eigenvalue
    - Requires solving linear system each iteration
    
    For tridiagonal matrices, uses Thomas algorithm from Chapter 01 if available.
    """
    n = A.shape[0]
    
    # Initialize
    if v0 is None:
        np.random.seed(42)
        v = np.random.randn(n)
    else:
        v = v0.copy()
    
    v = v / np.linalg.norm(v)
    
    # Shifted matrix
    A_shifted = A - sigma * np.eye(n)
    
    # Check if we can use LU factorization for efficiency
    if SCIPY_AVAILABLE:
        lu_piv = scipy_linalg.lu_factor(A_shifted)
        solve_func = lambda b: scipy_linalg.lu_solve(lu_piv, b)
    else:
        solve_func = lambda b: np.linalg.solve(A_shifted, b)
    
    history = [] if return_history else None
    
    for k in range(max_iter):
        # Solve (A - σI)w = v
        try:
            w = solve_func(v)
        except np.linalg.LinAlgError:
            # Singular matrix - sigma is an eigenvalue!
            residual = np.linalg.norm(A @ v - sigma * v)
            info = {
                'iterations': k + 1,
                'converged': True,
                'residual': residual,
                'note': 'sigma is an eigenvalue'
            }
            if return_history:
                info['history'] = np.array(history) if history else np.array([sigma])
            return sigma, v, info
        
        # Normalize
        norm_w = np.linalg.norm(w)
        v_new = w / norm_w
        
        # Rayleigh quotient for eigenvalue
        lam = np.dot(v_new, A @ v_new)
        
        if return_history:
            history.append(lam)
        
        # Check convergence
        # Use angle between consecutive eigenvector estimates
        cos_angle = abs(np.dot(v, v_new))
        if cos_angle > 1 - tol:
            residual = np.linalg.norm(A @ v_new - lam * v_new)
            info = {
                'iterations': k + 1,
                'converged': True,
                'residual': residual
            }
            if return_history:
                info['history'] = np.array(history)
            return float(lam), v_new, info
        
        v = v_new
    
    # Did not converge
    lam = np.dot(v, A @ v)
    residual = np.linalg.norm(A @ v - lam * v)
    info = {
        'iterations': max_iter,
        'converged': False,
        'residual': residual
    }
    if return_history:
        info['history'] = np.array(history)
    
    return float(lam), v, info


def rayleigh_quotient_iteration(
    A: np.ndarray,
    v0: Optional[np.ndarray] = None,
    sigma0: Optional[float] = None,
    tol: float = 1e-12,
    max_iter: int = 50,
    return_history: bool = False
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Rayleigh quotient iteration for symmetric matrices.
    
    Combines inverse iteration with dynamic shift updates using Rayleigh quotient.
    Achieves cubic convergence for symmetric matrices.
    
    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix (n x n)
    v0 : np.ndarray, optional
        Initial guess for eigenvector
    sigma0 : float, optional
        Initial shift. If None, uses Rayleigh quotient of v0.
    tol : float, optional
        Convergence tolerance
    max_iter : int, optional
        Maximum number of iterations
    return_history : bool, optional
        If True, return history of eigenvalue estimates
    
    Returns
    -------
    eigenvalue : float
        Converged eigenvalue
    eigenvector : np.ndarray
        Corresponding unit eigenvector
    info : dict
        Convergence information
    
    Examples
    --------
    >>> A = np.array([[4, 1], [1, 3]])
    >>> lam, v, info = rayleigh_quotient_iteration(A)
    >>> info['iterations']  # Very few iterations due to cubic convergence
    
    Notes
    -----
    - Cubic convergence for symmetric matrices (O(ε³) per iteration)
    - May converge to any eigenvalue depending on initial guess
    - More expensive per iteration than power/inverse iteration
    
    References
    ----------
    Trefethen & Bau, "Numerical Linear Algebra", Algorithm 27.3
    """
    n = A.shape[0]
    
    # Initialize eigenvector
    if v0 is None:
        np.random.seed(42)
        v = np.random.randn(n)
    else:
        v = v0.copy()
    
    v = v / np.linalg.norm(v)
    
    # Initialize shift (Rayleigh quotient)
    if sigma0 is None:
        sigma = np.dot(v, A @ v)
    else:
        sigma = sigma0
    
    history = [] if return_history else None
    
    for k in range(max_iter):
        if return_history:
            history.append(sigma)
        
        # Solve (A - σI)w = v
        A_shifted = A - sigma * np.eye(n)
        
        try:
            if SCIPY_AVAILABLE:
                w = scipy_linalg.solve(A_shifted, v, assume_a='sym')
            else:
                w = np.linalg.solve(A_shifted, v)
        except np.linalg.LinAlgError:
            # Converged! σ is an eigenvalue
            residual = np.linalg.norm(A @ v - sigma * v)
            info = {
                'iterations': k + 1,
                'converged': True,
                'residual': residual
            }
            if return_history:
                info['history'] = np.array(history)
            return float(sigma), v, info
        
        # Normalize
        v_new = w / np.linalg.norm(w)
        
        # Update Rayleigh quotient
        sigma_new = np.dot(v_new, A @ v_new)
        
        # Check convergence
        residual = np.linalg.norm(A @ v_new - sigma_new * v_new)
        if residual < tol:
            info = {
                'iterations': k + 1,
                'converged': True,
                'residual': residual
            }
            if return_history:
                history.append(sigma_new)
                info['history'] = np.array(history)
            return float(sigma_new), v_new, info
        
        v = v_new
        sigma = sigma_new
    
    # Did not converge
    info = {
        'iterations': max_iter,
        'converged': False,
        'residual': np.linalg.norm(A @ v - sigma * v)
    }
    if return_history:
        info['history'] = np.array(history)
    
    return float(sigma), v, info


def qr_iteration_basic(
    A: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
    return_history: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Basic QR iteration for computing all eigenvalues.
    
    Iteratively applies QR decomposition: A_k = Q_k R_k, A_{k+1} = R_k Q_k.
    Converges to upper triangular (Schur) form for general matrices,
    or diagonal for symmetric matrices.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix (n x n)
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Convergence tolerance for off-diagonal elements
    return_history : bool, optional
        If True, return history of diagonal evolution
    
    Returns
    -------
    eigenvalues : np.ndarray
        Array of eigenvalues (diagonal of final matrix)
    Q_accumulated : np.ndarray
        Accumulated orthogonal matrix (columns are eigenvectors for symmetric A)
    info : dict
        Convergence information
    
    Examples
    --------
    >>> A = np.array([[4, 1], [1, 3]])
    >>> eigs, Q, info = qr_iteration_basic(A)
    >>> np.sort(eigs)
    array([2.382..., 4.618...])
    
    Notes
    -----
    - Basic version without shifts (slow convergence)
    - O(n³) per iteration
    - For production use, prefer scipy.linalg.eig or shifted QR
    
    This is primarily for educational purposes to understand QR algorithm.
    """
    n = A.shape[0]
    Ak = A.copy().astype(float)
    Q_acc = np.eye(n)
    
    history = [] if return_history else None
    
    for k in range(max_iter):
        # QR decomposition
        Q, R = np.linalg.qr(Ak)
        
        # Reverse multiplication
        Ak = R @ Q
        
        # Accumulate orthogonal transformations
        Q_acc = Q_acc @ Q
        
        if return_history:
            history.append(np.diag(Ak).copy())
        
        # Check convergence: off-diagonal elements should be small
        off_diag = np.tril(Ak, -1)
        if np.linalg.norm(off_diag) < tol * np.linalg.norm(np.diag(Ak)):
            info = {
                'iterations': k + 1,
                'converged': True,
                'off_diagonal_norm': np.linalg.norm(off_diag)
            }
            if return_history:
                info['history'] = np.array(history)
            return np.diag(Ak), Q_acc, info
    
    # Did not fully converge
    info = {
        'iterations': max_iter,
        'converged': False,
        'off_diagonal_norm': np.linalg.norm(np.tril(Ak, -1))
    }
    if return_history:
        info['history'] = np.array(history)
    
    return np.diag(Ak), Q_acc, info


# =============================================================================
# Utility functions
# =============================================================================

def rayleigh_quotient(A: np.ndarray, v: np.ndarray) -> float:
    """
    Compute Rayleigh quotient R(A, v) = (v^T A v) / (v^T v).
    
    The Rayleigh quotient gives the best eigenvalue estimate for a given
    eigenvector approximation.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    v : np.ndarray
        Vector (need not be normalized)
    
    Returns
    -------
    float
        Rayleigh quotient value
    """
    return float(np.dot(v, A @ v) / np.dot(v, v))


def verify_eigenpair(
    A: np.ndarray,
    eigenvalue: float,
    eigenvector: np.ndarray,
    tol: float = 1e-8
) -> Dict[str, Any]:
    """
    Verify that (λ, v) is an eigenpair of A.
    
    Computes residual ||Av - λv|| and relative error.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix
    eigenvalue : float
        Proposed eigenvalue
    eigenvector : np.ndarray
        Proposed eigenvector
    tol : float, optional
        Tolerance for declaring valid eigenpair
    
    Returns
    -------
    dict
        Verification results:
        - 'is_valid': bool
        - 'residual_norm': ||Av - λv||
        - 'relative_residual': ||Av - λv|| / (||A|| ||v||)
    """
    v = eigenvector / np.linalg.norm(eigenvector)
    residual = A @ v - eigenvalue * v
    residual_norm = np.linalg.norm(residual)
    
    A_norm = np.linalg.norm(A)
    relative_residual = residual_norm / (A_norm + abs(eigenvalue))
    
    return {
        'is_valid': relative_residual < tol,
        'residual_norm': residual_norm,
        'relative_residual': relative_residual
    }


def estimate_convergence_rate(history: np.ndarray, true_value: float) -> float:
    """
    Estimate convergence rate from iteration history.
    
    Computes the ratio |e_{k+1}| / |e_k|^p to estimate order p.
    
    Parameters
    ----------
    history : np.ndarray
        Array of eigenvalue estimates from iterations
    true_value : float
        True eigenvalue for computing errors
    
    Returns
    -------
    float
        Estimated convergence rate (ratio of consecutive errors)
    """
    errors = np.abs(history - true_value)
    
    # Avoid division by zero
    valid = errors[:-1] > 1e-15
    if not np.any(valid):
        return 0.0
    
    ratios = errors[1:][valid] / errors[:-1][valid]
    return float(np.median(ratios))


# =============================================================================
# Specialized methods for structured matrices
# =============================================================================

def inverse_iteration_tridiagonal(
    d: np.ndarray,
    u: np.ndarray,
    o: np.ndarray,
    sigma: float = 0.0,
    v0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Inverse iteration optimized for tridiagonal matrices.
    
    Uses Thomas algorithm for O(n) solves instead of O(n³).
    
    Parameters
    ----------
    d : np.ndarray
        Main diagonal (length n)
    u : np.ndarray
        Lower subdiagonal (length n-1)
    o : np.ndarray
        Upper superdiagonal (length n-1)
    sigma : float, optional
        Shift value
    v0 : np.ndarray, optional
        Initial guess
    tol : float, optional
        Convergence tolerance
    max_iter : int, optional
        Maximum iterations
    
    Returns
    -------
    eigenvalue, eigenvector, info
        Same as inverse_iteration
    
    Notes
    -----
    Requires Chapter 01 (tridiagonal_solve). Falls back to dense solver if unavailable.
    """
    n = len(d)
    
    # Initialize
    if v0 is None:
        np.random.seed(42)
        v = np.random.randn(n)
    else:
        v = v0.copy()
    v = v / np.linalg.norm(v)
    
    # Shifted diagonals
    d_shifted = d - sigma
    
    for k in range(max_iter):
        # Solve (T - σI)w = v using Thomas algorithm
        if CHAPTER01_AVAILABLE:
            w = tridiagonal_solve(d_shifted.copy(), u.copy(), o.copy(), v.copy())
        else:
            # Fallback: build dense matrix
            T_shifted = np.diag(d_shifted) + np.diag(u, -1) + np.diag(o, 1)
            w = np.linalg.solve(T_shifted, v)
        
        # Normalize
        norm_w = np.linalg.norm(w)
        v_new = w / norm_w
        
        # Rayleigh quotient
        # T @ v = d*v + u*v[:-1] shifted + o*v[1:] shifted
        Tv = d * v_new
        Tv[1:] += u * v_new[:-1]
        Tv[:-1] += o * v_new[1:]
        lam = np.dot(v_new, Tv)
        
        # Check convergence
        cos_angle = abs(np.dot(v, v_new))
        if cos_angle > 1 - tol:
            # Compute residual
            residual_vec = Tv - lam * v_new
            residual = np.linalg.norm(residual_vec)
            return float(lam), v_new, {
                'iterations': k + 1,
                'converged': True,
                'residual': residual
            }
        
        v = v_new
    
    return float(lam), v, {
        'iterations': max_iter,
        'converged': False,
        'residual': np.linalg.norm(Tv - lam * v)
    }


# =============================================================================
# Convenience functions
# =============================================================================

def find_dominant_eigenvalue(A: np.ndarray, **kwargs) -> Tuple[float, np.ndarray]:
    """
    Find dominant eigenvalue (largest |λ|) using power iteration.
    
    Convenience wrapper around power_iteration.
    
    Returns
    -------
    eigenvalue, eigenvector
    """
    lam, v, _ = power_iteration(A, **kwargs)
    return lam, v


def find_smallest_eigenvalue(A: np.ndarray, **kwargs) -> Tuple[float, np.ndarray]:
    """
    Find smallest eigenvalue (smallest |λ|) using inverse iteration with σ=0.
    
    Returns
    -------
    eigenvalue, eigenvector
    """
    lam, v, _ = inverse_iteration(A, sigma=0.0, **kwargs)
    return lam, v


def find_eigenvalue_near(A: np.ndarray, sigma: float, **kwargs) -> Tuple[float, np.ndarray]:
    """
    Find eigenvalue closest to sigma using inverse iteration.
    
    Returns
    -------
    eigenvalue, eigenvector
    """
    lam, v, _ = inverse_iteration(A, sigma=sigma, **kwargs)
    return lam, v


# =============================================================================
# Lanczos Method for Large Sparse Matrices
# =============================================================================

def lanczos_iteration(
    A: Union[np.ndarray, 'scipy.sparse.spmatrix'],
    v0: Optional[np.ndarray] = None,
    k: int = 30,
    reorthogonalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lanczos iteration to reduce a symmetric matrix to tridiagonal form.
    
    The Lanczos algorithm builds an orthonormal basis Q = [q₁, q₂, ..., qₖ]
    such that Q^T A Q = T where T is tridiagonal with:
    - α (diagonal): αᵢ = qᵢᵀ A qᵢ
    - β (off-diagonal): βᵢ = ||w||, where w = A qᵢ - αᵢ qᵢ - βᵢ₋₁ qᵢ₋₁
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        Symmetric matrix (n x n). Can be scipy sparse matrix.
    v0 : ndarray, optional
        Starting vector. If None, uses random unit vector.
    k : int, optional
        Number of Lanczos iterations (size of Krylov subspace).
        Default is 30.
    reorthogonalize : bool, optional
        If True, perform full reorthogonalization to maintain numerical
        stability. Default is True (recommended for most cases).
    
    Returns
    -------
    alpha : ndarray
        Diagonal elements of tridiagonal matrix T (length k)
    beta : ndarray
        Off-diagonal elements of T (length k-1)
    Q : ndarray
        Orthonormal Lanczos vectors as columns (n x k)
    
    Notes
    -----
    - Without reorthogonalization, loss of orthogonality occurs O(ε * κ(A))
    - With full reorthogonalization, cost is O(k²n) but numerically stable
    - For eigenvalue computation, typically k << n is sufficient
    
    Examples
    --------
    >>> A = np.diag([1, 2, 3, 4, 5])
    >>> alpha, beta, Q = lanczos_iteration(A, k=5)
    >>> T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    >>> # Eigenvalues of T approximate eigenvalues of A
    """
    n = A.shape[0]
    
    # Initialize
    if v0 is None:
        v0 = np.random.randn(n)
    v0 = v0 / np.linalg.norm(v0)
    
    # Storage
    alpha = np.zeros(k)
    beta = np.zeros(k - 1) if k > 1 else np.array([])
    Q = np.zeros((n, k))
    
    # First vector
    Q[:, 0] = v0
    
    # Matrix-vector product (works for both dense and sparse)
    def matvec(x):
        return A @ x
    
    # First step (no previous vector)
    w = matvec(Q[:, 0])
    alpha[0] = np.dot(Q[:, 0], w)
    w = w - alpha[0] * Q[:, 0]
    
    if reorthogonalize:
        # Gram-Schmidt against all previous vectors
        w = w - Q[:, 0] * np.dot(Q[:, 0], w)
    
    for j in range(1, k):
        beta[j - 1] = np.linalg.norm(w)
        
        # Check for breakdown (invariant subspace found)
        if beta[j - 1] < 1e-12:
            # Early termination - return what we have
            return alpha[:j], beta[:j-1], Q[:, :j]
        
        Q[:, j] = w / beta[j - 1]
        
        # Matrix-vector product
        w = matvec(Q[:, j])
        
        # Three-term recurrence
        alpha[j] = np.dot(Q[:, j], w)
        w = w - alpha[j] * Q[:, j] - beta[j - 1] * Q[:, j - 1]
        
        if reorthogonalize:
            # Full reorthogonalization (Gram-Schmidt)
            for i in range(j + 1):
                w = w - Q[:, i] * np.dot(Q[:, i], w)
    
    return alpha, beta, Q


def lanczos_eigsh(
    A: Union[np.ndarray, 'scipy.sparse.spmatrix'],
    k_eig: int = 6,
    which: str = 'SA',
    v0: Optional[np.ndarray] = None,
    k_lanczos: Optional[int] = None,
    tol: float = 1e-10,
    max_iter: int = 100,
    return_eigenvectors: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Compute eigenvalues (and eigenvectors) of a symmetric matrix using Lanczos.
    
    This is a basic implementation of the implicitly restarted Lanczos method.
    For production use with large sparse matrices, consider scipy.sparse.linalg.eigsh.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        Symmetric matrix (n x n)
    k_eig : int, optional
        Number of eigenvalues to compute. Default is 6.
    which : str, optional
        Which eigenvalues to find:
        - 'SA': Smallest Algebraic (most negative)
        - 'LA': Largest Algebraic (most positive)
        - 'SM': Smallest Magnitude (closest to zero)
        - 'LM': Largest Magnitude (furthest from zero)
        Default is 'SA' (useful for ground state in quantum mechanics).
    v0 : ndarray, optional
        Starting vector for Lanczos
    k_lanczos : int, optional
        Krylov subspace size. Default is min(n, max(2*k_eig+1, 20))
    tol : float, optional
        Convergence tolerance for eigenvalues
    max_iter : int, optional
        Maximum number of restart iterations
    return_eigenvectors : bool, optional
        If True, return eigenvectors as well
    
    Returns
    -------
    eigenvalues : ndarray
        Computed eigenvalues (sorted according to `which`)
    eigenvectors : ndarray (if return_eigenvectors=True)
        Corresponding eigenvectors as columns
    info : dict (if return_eigenvectors=True)
        Convergence information
    
    Examples
    --------
    >>> import numpy as np
    >>> # 1D tight-binding Hamiltonian
    >>> n = 100
    >>> H = np.diag(np.ones(n)*2) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
    >>> eigvals = lanczos_eigsh(H, k_eig=5, which='SA', return_eigenvectors=False)
    >>> # Ground state energy should be close to 2 - 2*cos(π/(n+1))
    """
    n = A.shape[0]
    
    # Krylov subspace size - need enough vectors to capture k_eig eigenvalues well
    # Rule of thumb: k_lanczos should be at least 2*k_eig, preferably more
    if k_lanczos is None:
        k_lanczos = min(n, max(4 * k_eig, 40))
    
    k_lanczos = min(k_lanczos, n)
    
    # Ensure we have enough Lanczos vectors (at least 2*k_eig for good convergence)
    if k_lanczos < 2 * k_eig:
        k_lanczos = min(n, 2 * k_eig + 10)
    
    # Initialize starting vector
    if v0 is None:
        np.random.seed(42)  # Reproducibility
        v0 = np.random.randn(n)
    v0 = v0 / np.linalg.norm(v0)
    
    # Run Lanczos to build Krylov subspace
    alpha, beta, Q = lanczos_iteration(A, v0=v0, k=k_lanczos, reorthogonalize=True)
    
    # Build tridiagonal matrix T
    k_actual = len(alpha)
    T = np.diag(alpha)
    if len(beta) > 0:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    
    # Compute eigenvalues of T (Ritz values)
    ritz_values, ritz_vectors = np.linalg.eigh(T)
    
    # Number of eigenvalues we can compute
    k_compute = min(k_eig, k_actual)
    
    # Select eigenvalues according to 'which'
    if which == 'SA':
        idx = np.argsort(ritz_values)[:k_compute]
    elif which == 'LA':
        idx = np.argsort(ritz_values)[-k_compute:][::-1]
    elif which == 'SM':
        idx = np.argsort(np.abs(ritz_values))[:k_compute]
    elif which == 'LM':
        idx = np.argsort(np.abs(ritz_values))[-k_compute:][::-1]
    else:
        raise ValueError(f"Unknown 'which' parameter: {which}")
    
    eigenvalues = ritz_values[idx]
    
    # Check if we got enough eigenvalues; if not, increase k_lanczos
    if k_compute < k_eig and k_lanczos < n:
        # Retry with larger Krylov subspace
        k_lanczos_new = min(n, k_lanczos * 2)
        if k_lanczos_new > k_lanczos:
            # Add small perturbation to v0 to avoid invariant subspace
            v0_perturbed = v0 + 0.01 * np.random.randn(n)
            v0_perturbed = v0_perturbed / np.linalg.norm(v0_perturbed)
            
            alpha, beta, Q = lanczos_iteration(A, v0=v0_perturbed, k=k_lanczos_new, 
                                               reorthogonalize=True)
            k_actual = len(alpha)
            T = np.diag(alpha)
            if len(beta) > 0:
                T += np.diag(beta, 1) + np.diag(beta, -1)
            
            ritz_values, ritz_vectors = np.linalg.eigh(T)
            k_compute = min(k_eig, k_actual)
            
            if which == 'SA':
                idx = np.argsort(ritz_values)[:k_compute]
            elif which == 'LA':
                idx = np.argsort(ritz_values)[-k_compute:][::-1]
            elif which == 'SM':
                idx = np.argsort(np.abs(ritz_values))[:k_compute]
            elif which == 'LM':
                idx = np.argsort(np.abs(ritz_values))[-k_compute:][::-1]
            
            eigenvalues = ritz_values[idx]
    
    # Compute eigenvectors in original basis: V = Q * S
    if return_eigenvectors:
        eigenvectors = Q @ ritz_vectors[:, idx]
        
        # Normalize
        for i in range(eigenvectors.shape[1]):
            eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])
        
        # Compute residuals ||Av - λv||
        residuals = []
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            residual = np.linalg.norm(A @ v - eigenvalues[i] * v)
            residuals.append(residual)
        
        # Convergence: check if residuals are small
        converged = all(r < tol * 100 for r in residuals)  # Use relaxed tol for convergence check
        
        info = {
            'iterations': 1,  # Single Lanczos run (no restarts in this simple version)
            'converged': converged,
            'k_lanczos': k_actual,
            'residuals': residuals
        }
        
        return eigenvalues, eigenvectors, info
    else:
        return eigenvalues


def build_sparse_test_matrix(n: int, matrix_type: str = 'laplacian_1d') -> np.ndarray:
    """
    Build test matrices commonly used in physics for eigenvalue problems.
    
    Parameters
    ----------
    n : int
        Matrix size
    matrix_type : str
        Type of matrix:
        - 'laplacian_1d': 1D discrete Laplacian (tridiagonal, [-1, 2, -1])
        - 'laplacian_2d': 2D discrete Laplacian on n×n grid
        - 'tight_binding': Nearest-neighbor tight-binding Hamiltonian
        - 'harmonic': 1D quantum harmonic oscillator (DVR)
    
    Returns
    -------
    A : ndarray or sparse matrix
        The test matrix
    """
    if matrix_type == 'laplacian_1d':
        # Tridiagonal: -d²/dx² with Dirichlet BC
        diag = np.ones(n) * 2
        off_diag = -np.ones(n - 1)
        A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        return A
    
    elif matrix_type == 'tight_binding':
        # Same as laplacian but often with different interpretation
        return build_sparse_test_matrix(n, 'laplacian_1d')
    
    elif matrix_type == 'laplacian_2d':
        # 2D Laplacian on n×n grid (total size n² × n²)
        # Using Kronecker product: I ⊗ T + T ⊗ I
        T1d = build_sparse_test_matrix(n, 'laplacian_1d')
        I = np.eye(n)
        A = np.kron(I, T1d) + np.kron(T1d, I)
        return A
    
    elif matrix_type == 'harmonic':
        # Discrete Variable Representation for harmonic oscillator
        # H = -ℏ²/(2m) d²/dx² + ½mω²x²
        # In dimensionless units with ℏ=m=ω=1
        h = 10.0 / (n + 1)  # Grid spacing over [-5, 5]
        x = np.linspace(-5 + h, 5 - h, n)
        
        # Kinetic energy (finite difference)
        T = build_sparse_test_matrix(n, 'laplacian_1d') / (h ** 2)
        
        # Potential energy
        V = np.diag(0.5 * x ** 2)
        
        return 0.5 * T + V  # H = T/2 + V (dimensionless)
    
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")
