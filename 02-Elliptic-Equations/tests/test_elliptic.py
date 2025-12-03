import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import scipy.sparse as sp
from elliptic import build_poisson_2d, solve_direct, jacobi


def test_build_small():
    nx, ny = 6, 5
    bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
          'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 1.0)}
    A, b, meta = build_poisson_2d(nx, ny, lx=1.0, ly=1.0, bc=bc)
    assert A.shape[0] == (nx-2)*(ny-2)
    # matrix should be sparse
    assert sp.issparse(A)


def test_direct_vs_iterative():
    nx, ny = 20, 20
    bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
          'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 1.0)}
    A, b, meta = build_poisson_2d(nx, ny, lx=1.0, ly=1.0, bc=bc)
    x_direct = solve_direct(A, b)
    x_j, it, r = jacobi(A, b, tol=1e-6, maxiter=5000)

    # check relative difference small
    diff = np.linalg.norm(x_direct - x_j) / np.linalg.norm(x_direct)
    assert diff < 1e-1  # iterative limited tolerance; check rough agreement
