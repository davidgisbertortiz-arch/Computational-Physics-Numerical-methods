"""
Linear Systems Module

This module provides functions for working with linear systems,
particularly tridiagonal systems that arise in computational physics.
"""

import numpy as np
from typing import Tuple, Optional


def build_tridiagonal(d: np.ndarray, u: np.ndarray, o: np.ndarray) -> np.ndarray:
    """
    Build a dense tridiagonal matrix from three diagonal arrays.
    
    Parameters
    ----------
    d : np.ndarray
        Main diagonal (length n)
    u : np.ndarray
        Lower subdiagonal (length n-1)
    o : np.ndarray
        Upper superdiagonal (length n-1)
    
    Returns
    -------
    np.ndarray
        Dense n×n tridiagonal matrix
    
    Examples
    --------
    >>> d = np.array([2, 2, 2])
    >>> u = np.array([-1, -1])
    >>> o = np.array([-1, -1])
    >>> A = build_tridiagonal(d, u, o)
    """
    n = len(d)
    A = np.zeros((n, n))
    
    # Main diagonal
    np.fill_diagonal(A, d)
    
    # Lower subdiagonal
    np.fill_diagonal(A[1:], u)
    
    # Upper superdiagonal
    np.fill_diagonal(A[:, 1:], o)
    
    return A


def build_discrete_laplacian_1d(n: int, h: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build tridiagonal matrix for discrete 1D Laplacian (second derivative).
    
    Approximates d²u/dx² on interval [0,1] with n interior points.
    Uses finite difference: (u_{i-1} - 2u_i + u_{i+1}) / h²
    
    Parameters
    ----------
    n : int
        Number of interior points
    h : float, optional
        Grid spacing. If None, computed as 1/(n+1)
    
    Returns
    -------
    tuple of (d, u, o)
        Main diagonal, subdiagonal, and superdiagonal arrays
    """
    if h is None:
        h = 1.0 / (n + 1)
    
    d = np.full(n, -2.0 / h**2)
    u = np.full(n - 1, 1.0 / h**2)
    o = np.full(n - 1, 1.0 / h**2)
    
    return d, u, o


def tridiagonal_solve(d: np.ndarray, u: np.ndarray, o: np.ndarray, 
                      b: np.ndarray, modify_inplace: bool = False) -> np.ndarray:
    """
    Solve tridiagonal system using Thomas algorithm.
    
    Solves Ax = b where A is tridiagonal with:
    - d: main diagonal
    - u: lower subdiagonal
    - o: upper superdiagonal
    
    Time complexity: O(n)
    Space complexity: O(n)
    
    Parameters
    ----------
    d : np.ndarray
        Main diagonal (length n)
    u : np.ndarray
        Lower subdiagonal (length n-1)
    o : np.ndarray
        Upper superdiagonal (length n-1)
    b : np.ndarray
        Right-hand side vector (length n)
    modify_inplace : bool, optional
        If True, modifies input arrays. If False, works on copies.
    
    Returns
    -------
    np.ndarray
        Solution vector x
    
    Notes
    -----
    The Thomas algorithm consists of:
    1. Forward elimination to create upper triangular system
    2. Back substitution to solve for x
    """
    n = len(d)
    
    if not modify_inplace:
        d = d.copy()
        o = o.copy()
        b = b.copy()
    
    # Forward elimination
    for i in range(1, n):
        factor = u[i - 1] / d[i - 1]
        d[i] -= factor * o[i - 1]
        b[i] -= factor * b[i - 1]
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = b[-1] / d[-1]
    
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - o[i] * x[i + 1]) / d[i]
    
    return x


def create_random_system(n: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random dense linear system Ax = b.
    
    Parameters
    ----------
    n : int
        System size
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    tuple of (A, b)
        Random matrix A and vector b
    """
    if seed is not None:
        np.random.seed(seed)
    
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    
    return A, b


def compute_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the L2 norm of the residual ||Ax - b||₂.
    
    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix
    x : np.ndarray
        Solution vector
    b : np.ndarray
        Right-hand side vector
    
    Returns
    -------
    float
        L2 norm of residual
    """
    return np.linalg.norm(A @ x - b)


def compute_relative_error(x_computed: np.ndarray, x_exact: np.ndarray) -> float:
    """
    Compute relative error ||x_computed - x_exact||₂ / ||x_exact||₂.
    
    Parameters
    ----------
    x_computed : np.ndarray
        Computed solution
    x_exact : np.ndarray
        Exact solution
    
    Returns
    -------
    float
        Relative error
    """
    return np.linalg.norm(x_computed - x_exact) / np.linalg.norm(x_exact)
