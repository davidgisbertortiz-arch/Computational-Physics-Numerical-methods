"""
Tests for multigrid methods in elliptic.py

Tests cover:
- Restriction and prolongation operators
- Residual computation
- Red-black Gauss-Seidel smoother
- V-cycle convergence
- Full multigrid solver
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from elliptic import (
    restrict_injection, restrict_full_weighting, prolong_linear,
    residual_2d, smooth_gauss_seidel_rb, v_cycle, multigrid_solve,
    build_poisson_2d
)


def test_restriction_sizes():
    """Test that restriction operators produce correct output sizes."""
    u_fine = np.random.randn(17, 17)
    
    # Injection
    u_coarse_inj = restrict_injection(u_fine)
    assert u_coarse_inj.shape == (9, 9), "Injection: 17x17 -> 9x9"
    
    # Full weighting
    u_coarse_fw = restrict_full_weighting(u_fine)
    assert u_coarse_fw.shape == (9, 9), "Full weighting: 17x17 -> 9x9"


def test_prolongation_sizes():
    """Test that prolongation produces correct output sizes."""
    u_coarse = np.random.randn(9, 9)
    u_fine = prolong_linear(u_coarse, 17, 17)
    assert u_fine.shape == (17, 17), "Prolongation: 9x9 -> 17x17"


def test_restrict_prolong_identity():
    """Test that restrict->prolong preserves smooth functions reasonably."""
    n_fine = 17
    x = np.linspace(0, 1, n_fine)
    y = np.linspace(0, 1, n_fine)
    X, Y = np.meshgrid(x, y)
    
    # Smooth test function
    u_original = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Restrict and prolong
    u_coarse = restrict_full_weighting(u_original)
    u_back = prolong_linear(u_coarse, n_fine, n_fine)
    
    # Should be similar (not exact due to downsampling)
    rel_error = np.linalg.norm(u_original - u_back) / np.linalg.norm(u_original)
    assert rel_error < 0.1, f"Restrict->Prolong error too large: {rel_error}"


def test_residual_zero_solution():
    """Test that residual is zero for exact solution."""
    n = 8
    h = 1.0 / (n + 1)
    
    # Homogeneous problem: f=0, u=0 (exact solution)
    u = np.zeros((n, n))
    f = np.zeros((n, n))
    
    r = residual_2d(u, f, h, h)
    
    # Residual should be zero (except at boundaries which are ignored)
    assert np.max(np.abs(r[1:-1, 1:-1])) < 1e-10, "Residual should be zero for exact solution"


def test_residual_constant_solution():
    """Test residual for constant solution (Laplacian = 0)."""
    n = 8
    h = 1.0 / (n + 1)
    
    # Constant function u=c has Laplacian=0
    u = np.ones((n, n)) * 5.0
    f = np.zeros((n, n))
    
    r = residual_2d(u, f, h, h)
    
    # Interior residual should be zero
    assert np.max(np.abs(r[1:-1, 1:-1])) < 1e-10


def test_smoother_reduces_error():
    """Test that Gauss-Seidel smoother reduces residual."""
    n = 16
    h = 1.0 / (n + 1)
    
    # Setup problem
    u = np.random.randn(n, n) * 0.1
    f = np.zeros((n, n))
    
    # Apply boundaries (Dirichlet = 0)
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    
    # Initial residual
    r0 = np.linalg.norm(residual_2d(u, f, h, h))
    
    # Smooth
    u_smooth = smooth_gauss_seidel_rb(u, f, h, h, iterations=5)
    r1 = np.linalg.norm(residual_2d(u_smooth, f, h, h))
    
    # Residual should decrease
    assert r1 < r0, "Smoother should reduce residual"
    assert r1 < 0.5 * r0, "Smoother should significantly reduce residual"


def test_v_cycle_convergence():
    """Test that V-cycle reduces residual."""
    n = 15
    h = 1.0 / (n + 1)
    
    # Setup problem: -∇²u = 0 with BC
    u = np.zeros((n+2, n+2))
    u[-1, :] = 1.0  # Top boundary
    f = np.zeros((n+2, n+2))
    
    # Initial residual
    r0 = np.linalg.norm(residual_2d(u, f, h, h))
    
    # One V-cycle
    u_improved = v_cycle(u.copy(), f, h, h, levels=3, pre_smooth=2, post_smooth=2)
    r1 = np.linalg.norm(residual_2d(u_improved, f, h, h))
    
    # Should reduce residual significantly
    assert r1 < r0, "V-cycle should reduce residual"
    assert r1 < 0.5 * r0, "V-cycle should significantly reduce residual"


def test_multigrid_small_problem():
    """Test multigrid solver on small problem."""
    nx = ny = 15
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    
    u, iterations, hist = multigrid_solve(nx, ny, bc=bc, tol=1e-6, maxiter=50, verbose=False)
    
    # Should converge
    assert iterations < 50, f"Should converge in < 50 iterations, got {iterations}"
    assert iterations < 20, f"Should converge quickly, got {iterations} iterations"
    
    # Solution should satisfy boundary conditions
    assert np.allclose(u[0, :], 0), "Bottom BC not satisfied"
    assert np.allclose(u[-1, :], 1), "Top BC not satisfied"
    assert np.allclose(u[:, 0], 0), "Left BC not satisfied"
    assert np.allclose(u[:, -1], 0), "Right BC not satisfied"
    
    # Solution should be smooth (monotonic in y for this problem)
    mid_slice = u[:, nx//2 + 1]
    assert np.all(np.diff(mid_slice) >= -1e-6), "Solution should be monotonic"


def test_multigrid_convergence_history():
    """Test that multigrid convergence history shows monotonic decrease."""
    nx = ny = 31
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    
    u, iterations, hist = multigrid_solve(nx, ny, bc=bc, tol=1e-8, maxiter=30, verbose=False)
    
    # History should be monotonically decreasing
    hist_array = np.array(hist)
    assert np.all(np.diff(hist_array) <= 1e-10), "Residual history should decrease"
    
    # Final residual should be small
    assert hist[-1] / hist[0] < 1e-6, "Relative residual should be < 1e-6"


def test_multigrid_vs_direct():
    """Compare multigrid solution with direct solver."""
    nx = ny = 31
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    
    # Multigrid
    u_mg, _, _ = multigrid_solve(nx, ny, bc=bc, tol=1e-8, verbose=False)
    
    # Direct
    from elliptic import solve_direct
    A, b, meta = build_poisson_2d(nx, ny, bc=bc)
    x_direct = solve_direct(A, b)
    
    # Extract interior points from multigrid solution
    u_mg_interior = u_mg[1:-1, 1:-1].flatten()
    
    # Compare
    rel_error = np.linalg.norm(u_mg_interior - x_direct) / np.linalg.norm(x_direct)
    assert rel_error < 1e-5, f"Multigrid solution differs from direct: {rel_error}"


def test_multigrid_grid_independent_convergence():
    """Test that multigrid iterations are approximately grid-independent."""
    sizes = [15, 31, 63]
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    
    iterations_list = []
    for n in sizes:
        _, iters, _ = multigrid_solve(n, n, bc=bc, tol=1e-8, maxiter=30, verbose=False)
        iterations_list.append(iters)
    
    # Iterations should not increase significantly with grid size
    max_iters = max(iterations_list)
    min_iters = min(iterations_list)
    
    print(f"Iterations for sizes {sizes}: {iterations_list}")
    assert max_iters - min_iters < 5, "Iterations should be grid-independent"


def test_multigrid_with_source():
    """Test multigrid with non-zero source term."""
    nx = ny = 31
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 0)}
    
    # Source term: f = 1
    source = np.ones((ny, nx))
    
    u, iters, hist = multigrid_solve(nx, ny, bc=bc, source_term=source, 
                                     tol=1e-8, maxiter=30, verbose=False)
    
    # Should converge
    assert iters < 30, f"Should converge with source term, got {iters} iterations"
    
    # Solution should be positive in interior (u=0 on boundary, f>0 interior)
    assert np.all(u[1:-1, 1:-1] > -1e-6), "Solution should be non-negative with f>0"


@pytest.mark.parametrize("grid_size", [15, 31, 63])
def test_multigrid_different_sizes(grid_size):
    """Parametrized test for different grid sizes."""
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    
    u, iters, hist = multigrid_solve(grid_size, grid_size, bc=bc, 
                                     tol=1e-6, maxiter=50, verbose=False)
    
    assert iters < 50, f"Should converge for size {grid_size}"
    assert u.shape == (grid_size + 2, grid_size + 2), "Output size incorrect"
    
    # Check boundary conditions
    assert np.allclose(u[-1, :], 1, atol=1e-6), f"Top BC failed for size {grid_size}"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
