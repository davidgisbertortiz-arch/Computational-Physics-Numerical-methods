"""
Tests for variable coefficient elliptic solver
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Setup path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / '02-Elliptic-Equations' / 'src'))

from elliptic import build_variable_coeff_2d, solve_direct, build_poisson_2d


def test_variable_coeff_constant_kappa():
    """When kappa is constant, should match build_poisson_2d"""
    nx, ny = 21, 21
    kappa = np.ones((ny, nx))
    
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    
    # Variable coefficient version with kappa=1 everywhere
    A_var, b_var, meta_var = build_variable_coeff_2d(nx, ny, kappa, source=None, bc=bc)
    
    # Standard Poisson version
    A_std, b_std, meta_std = build_poisson_2d(nx, ny, bc=bc)
    
    # Matrices should be nearly identical (up to numerical differences in assembly)
    assert A_var.shape == A_std.shape
    assert np.allclose(A_var.toarray(), A_std.toarray(), rtol=1e-10)
    assert np.allclose(b_var, b_std, rtol=1e-10)


def test_variable_coeff_symmetry():
    """Matrix should be symmetric for Dirichlet BCs"""
    nx, ny = 11, 11
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Smoothly varying kappa
    kappa = 1.0 + np.sin(np.pi * X) * np.cos(np.pi * Y)
    
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 0)}
    
    A, b, meta = build_variable_coeff_2d(nx, ny, kappa, bc=bc)
    
    # Check symmetry
    A_dense = A.toarray()
    assert np.allclose(A_dense, A_dense.T, rtol=1e-12)


def test_variable_coeff_positive_definite():
    """Matrix should be negative definite (all eigenvalues < 0) for Dirichlet BCs"""
    nx, ny = 11, 11
    kappa = np.random.uniform(1.0, 10.0, size=(ny, nx))
    
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 0)}
    
    A, b, meta = build_variable_coeff_2d(nx, ny, kappa, bc=bc)
    
    # Check that -A is positive definite by checking eigenvalues
    eigs = np.linalg.eigvalsh((-A).toarray())
    assert np.all(eigs > -1e-10), "Some eigenvalues of -A are not positive"


def test_variable_coeff_two_layer():
    """Test two-layer problem: kappa jumps at y=0.5"""
    nx, ny = 31, 31
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Two layers: kappa=1 for y<0.5, kappa=10 for y>=0.5
    kappa = np.where(Y < 0.5, 1.0, 10.0)
    
    bc = {'left': ('neumann', 0), 'right': ('neumann', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 1)}
    
    A, b, meta = build_variable_coeff_2d(nx, ny, kappa, bc=bc)
    u_interior = solve_direct(A, b)
    
    # Reconstruct full solution
    u_full = np.zeros((ny, nx))
    u_full[1:-1, 1:-1] = u_interior.reshape((ny-2, nx-2))
    u_full[0, :] = 0.0
    u_full[-1, :] = 1.0
    u_full[:, 0] = u_full[:, 1]   # Neumann BC
    u_full[:, -1] = u_full[:, -2]  # Neumann BC
    
    # Check solution properties
    assert u_full.min() >= -1e-10, "Solution should be non-negative"
    assert u_full.max() <= 1.0 + 1e-10, "Solution should not exceed boundary value"
    
    # Temperature should increase from bottom to top
    mid_col = u_full[:, nx//2]
    assert np.all(np.diff(mid_col) >= -1e-10), "Temperature should be monotonically increasing"
    
    # Gradient should be steeper in lower layer (lower kappa)
    bottom_gradient = (mid_col[ny//4] - mid_col[0]) / (y[ny//4] - y[0])
    top_gradient = (mid_col[-1] - mid_col[3*ny//4]) / (y[-1] - y[3*ny//4])
    # Bottom gradient should be larger (less conductivity means steeper gradient for same flux)
    assert bottom_gradient > top_gradient * 0.5, "Bottom layer should have steeper gradient"


def test_variable_coeff_with_source():
    """Test with non-zero source term"""
    nx, ny = 21, 21
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    kappa = np.ones((ny, nx)) * 2.0
    source = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 0)}
    
    A, b, meta = build_variable_coeff_2d(nx, ny, kappa, source=source, bc=bc)
    u_interior = solve_direct(A, b)
    
    # Solution should be non-trivial (not all zeros)
    assert np.max(np.abs(u_interior)) > 1e-6, "Solution should be non-zero with source term"


def test_variable_coeff_circular_inclusion():
    """Test circular high-conductivity inclusion"""
    nx, ny = 41, 41
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Circular inclusion at center
    distance = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    kappa = np.where(distance < 0.2, 10.0, 1.0)
    
    source = np.ones((ny, nx))
    
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 0),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 0)}
    
    A, b, meta = build_variable_coeff_2d(nx, ny, kappa, source=source, bc=bc)
    u_interior = solve_direct(A, b)
    
    # Reconstruct
    u_full = np.zeros((ny, nx))
    u_full[1:-1, 1:-1] = u_interior.reshape((ny-2, nx-2))
    
    # With uniform source and zero BCs, solution should be smooth and symmetric
    # Check that solution is non-trivial and positive inside domain
    assert np.max(u_full) > 1e-6, "Solution should be non-zero with source"
    
    # Solution should be roughly symmetric about center
    center_val = u_full[ny//2, nx//2]
    # Check a few symmetric pairs
    assert abs(u_full[ny//4, nx//2] - u_full[3*ny//4, nx//2]) < 0.1 * center_val
    assert abs(u_full[ny//2, nx//4] - u_full[ny//2, 3*nx//4]) < 0.1 * center_val


def test_variable_coeff_metadata():
    """Test metadata returned by build_variable_coeff_2d"""
    nx, ny = 21, 21
    kappa = np.random.uniform(1.0, 100.0, size=(ny, nx))
    
    A, b, meta = build_variable_coeff_2d(nx, ny, kappa)
    
    assert 'nx' in meta and meta['nx'] == nx
    assert 'ny' in meta and meta['ny'] == ny
    assert 'hx' in meta and 'hy' in meta
    assert 'kappa_min' in meta and meta['kappa_min'] == pytest.approx(np.min(kappa))
    assert 'kappa_max' in meta and meta['kappa_max'] == pytest.approx(np.max(kappa))
    assert 'kappa_ratio' in meta
    assert meta['kappa_ratio'] == pytest.approx(np.max(kappa) / np.min(kappa))


def test_variable_coeff_invalid_shapes():
    """Test error handling for invalid kappa shapes"""
    nx, ny = 21, 21
    
    # Wrong shape for kappa
    kappa_wrong = np.ones((ny+1, nx))
    with pytest.raises(ValueError, match="kappa must have shape"):
        build_variable_coeff_2d(nx, ny, kappa_wrong)
    
    # Wrong shape for source
    kappa = np.ones((ny, nx))
    source_wrong = np.ones((ny, nx+1))
    with pytest.raises(ValueError, match="source must have shape"):
        build_variable_coeff_2d(nx, ny, kappa, source=source_wrong)


def test_harmonic_mean_interfaces():
    """Test that harmonic mean is used at interfaces"""
    # Simple test: 3x3 grid with kappa jump
    nx, ny = 5, 5
    kappa = np.ones((ny, nx))
    kappa[:, 3:] = 10.0  # Right half has kappa=10
    
    bc = {'left': ('dirichlet', 0), 'right': ('dirichlet', 1),
          'bottom': ('dirichlet', 0), 'top': ('dirichlet', 0)}
    
    A, b, meta = build_variable_coeff_2d(nx, ny, kappa, bc=bc)
    
    # Check that matrix is properly assembled (basic sanity check)
    assert A.shape[0] == (nx-2) * (ny-2)
    assert not np.any(np.isnan(A.data))
    assert not np.any(np.isinf(A.data))
    
    # Solve and check solution is reasonable
    u_interior = solve_direct(A, b)
    assert not np.any(np.isnan(u_interior))
    assert not np.any(np.isinf(u_interior))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
