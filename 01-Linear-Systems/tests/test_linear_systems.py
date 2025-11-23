"""
Unit tests for linear_systems module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from linear_systems import (
    build_tridiagonal,
    build_discrete_laplacian_1d,
    tridiagonal_solve,
    compute_residual,
    compute_relative_error
)


class TestBuildTridiagonal:
    def test_simple_tridiagonal(self):
        """Test building a simple 3x3 tridiagonal matrix."""
        d = np.array([2.0, 2.0, 2.0])
        u = np.array([-1.0, -1.0])
        o = np.array([-1.0, -1.0])
        
        A = build_tridiagonal(d, u, o)
        
        expected = np.array([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0]
        ])
        
        assert np.allclose(A, expected)
    
    def test_matrix_shape(self):
        """Test that output matrix has correct shape."""
        n = 10
        d = np.ones(n)
        u = np.ones(n - 1)
        o = np.ones(n - 1)
        
        A = build_tridiagonal(d, u, o)
        
        assert A.shape == (n, n)


class TestDiscreteLaplacian:
    def test_laplacian_structure(self):
        """Test discrete Laplacian has correct structure."""
        n = 5
        d, u, o = build_discrete_laplacian_1d(n)
        A = build_tridiagonal(d, u, o)
        
        # Should be symmetric
        assert np.allclose(A, A.T)
        
        # Should have correct dimensions
        assert len(d) == n
        assert len(u) == n - 1
        assert len(o) == n - 1
    
    def test_laplacian_eigenvalues(self):
        """Test that Laplacian has negative eigenvalues."""
        n = 10
        d, u, o = build_discrete_laplacian_1d(n)
        A = build_tridiagonal(d, u, o)
        
        eigenvalues = np.linalg.eigvalsh(A)
        
        # All eigenvalues should be negative
        assert np.all(eigenvalues < 0)


class TestTridiagonalSolve:
    def test_known_solution(self):
        """Test solver with known solution."""
        n = 5
        d, u, o = build_discrete_laplacian_1d(n)
        
        # Create exact solution and compute b
        x_exact = np.ones(n)
        A = build_tridiagonal(d, u, o)
        b = A @ x_exact
        
        # Solve
        x_computed = tridiagonal_solve(d, u, o, b)
        
        # Check solution
        assert np.allclose(x_computed, x_exact, rtol=1e-10)
    
    def test_comparison_with_numpy(self):
        """Test that Thomas algorithm matches numpy.linalg.solve."""
        n = 20
        d, u, o = build_discrete_laplacian_1d(n)
        b = np.random.randn(n)
        
        # Solve with Thomas algorithm
        x_thomas = tridiagonal_solve(d, u, o, b)
        
        # Solve with NumPy
        A = build_tridiagonal(d, u, o)
        x_numpy = np.linalg.solve(A, b)
        
        # Solutions should be very close
        assert np.allclose(x_thomas, x_numpy, rtol=1e-10)
    
    def test_residual_small(self):
        """Test that residual is small."""
        n = 100
        d, u, o = build_discrete_laplacian_1d(n)
        b = np.random.randn(n)
        
        x = tridiagonal_solve(d, u, o, b)
        A = build_tridiagonal(d, u, o)
        residual = compute_residual(A, x, b)
        
        # Residual should be very small
        assert residual < 1e-10
    
    def test_no_modification_when_copy(self):
        """Test that input arrays are not modified when modify_inplace=False."""
        n = 10
        d, u, o = build_discrete_laplacian_1d(n)
        b = np.random.randn(n)
        
        d_orig = d.copy()
        o_orig = o.copy()
        b_orig = b.copy()
        
        _ = tridiagonal_solve(d, u, o, b, modify_inplace=False)
        
        assert np.allclose(d, d_orig)
        assert np.allclose(o, o_orig)
        assert np.allclose(b, b_orig)


class TestUtilityFunctions:
    def test_compute_residual(self):
        """Test residual computation."""
        A = np.array([[2, -1], [-1, 2]])
        x = np.array([1, 1])
        b = A @ x
        
        residual = compute_residual(A, x, b)
        
        assert residual < 1e-14
    
    def test_compute_relative_error(self):
        """Test relative error computation."""
        x_exact = np.array([1.0, 2.0, 3.0])
        x_computed = np.array([1.01, 2.02, 3.03])
        
        rel_error = compute_relative_error(x_computed, x_exact)
        
        expected_error = 0.01 * np.sqrt(1 + 4 + 9) / np.sqrt(1 + 4 + 9)
        assert np.isclose(rel_error, expected_error)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
