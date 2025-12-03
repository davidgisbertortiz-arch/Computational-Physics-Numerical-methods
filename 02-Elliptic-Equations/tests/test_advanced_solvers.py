import sys
from pathlib import Path

# Add both chapter paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / '02-Elliptic-Equations' / 'src'))
sys.path.insert(0, str(repo_root / '01-Linear-Systems' / 'src'))

import numpy as np
import scipy.sparse as sp

from elliptic import (build_poisson_2d, solve_direct, line_relaxation, 
                      adi_solve, CHAPTER01_AVAILABLE)
from linear_systems import compute_residual, compute_relative_error


def test_line_relaxation_convergence():
    """Test that line relaxation converges to the direct solution."""
    if not CHAPTER01_AVAILABLE:
        print('Skipping: Chapter 01 not available')
        return
    
    nx, ny = 30, 20
    bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
          'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 1.0)}
    A, b, meta = build_poisson_2d(nx, ny, lx=1.0, ly=1.0, bc=bc)
    nx_tot, ny_tot, hx, hy = meta
    nx_i = nx_tot - 2
    ny_i = ny_tot - 2
    
    x_direct = solve_direct(A, b)
    x_line, it, res = line_relaxation(nx_i, ny_i, hx, hy, bc, b, tol=1e-6, maxiter=500)
    
    rel_err = compute_relative_error(x_line, x_direct)
    assert rel_err < 1e-4, f'Line relaxation error too high: {rel_err}'
    print(f'Line relaxation: {it} iters, rel_err={rel_err:.3e}')


def test_adi_convergence():
    """Test that ADI method converges to the direct solution."""
    if not CHAPTER01_AVAILABLE:
        print('Skipping: Chapter 01 not available')
        return
    
    nx, ny = 30, 20
    bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
          'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 1.0)}
    A, b, meta = build_poisson_2d(nx, ny, lx=1.0, ly=1.0, bc=bc)
    nx_tot, ny_tot, hx, hy = meta
    nx_i = nx_tot - 2
    ny_i = ny_tot - 2
    
    x_direct = solve_direct(A, b)
    x_adi, it, res = adi_solve(nx_i, ny_i, hx, hy, bc, b, tol=1e-6, maxiter=300)
    
    rel_err = compute_relative_error(x_adi, x_direct)
    assert rel_err < 1e-4, f'ADI error too high: {rel_err}'
    print(f'ADI: {it} iters, rel_err={rel_err:.3e}')


def test_residual_computation():
    """Test that chapter 01 compute_residual works with chapter 02 problems."""
    if not CHAPTER01_AVAILABLE:
        print('Skipping: Chapter 01 not available')
        return
    
    nx, ny = 20, 15
    bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
          'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 1.0)}
    A, b, meta = build_poisson_2d(nx, ny, lx=1.0, ly=1.0, bc=bc)
    
    x = solve_direct(A, b)
    residual = compute_residual(A.toarray(), x, b)
    assert residual < 1e-10, f'Residual too high for direct solve: {residual}'
    print(f'Residual for direct solve: {residual:.3e}')


if __name__ == '__main__':
    print('Testing advanced solvers...')
    test_line_relaxation_convergence()
    test_adi_convergence()
    test_residual_computation()
    print('All advanced solver tests passed!')
