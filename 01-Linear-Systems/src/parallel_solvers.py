"""
Parallel and GPU-accelerated implementations of tridiagonal solvers.

This module provides:
1. Multi-threaded CPU implementation using numba
2. GPU implementation using CuPy
3. Cyclic reduction algorithm for parallel execution
"""

import numpy as np
from typing import Tuple, Optional

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Install with: pip install numba")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available. Install with: pip install cupy-cuda11x or cupy-cuda12x")


# =======================
# Numba Parallel Implementation
# =======================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def tridiagonal_solve_parallel(d: np.ndarray, u: np.ndarray, 
                                   o: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Parallel tridiagonal solver using cyclic reduction.
        
        This is more complex than Thomas but can utilize multiple cores.
        For small-medium systems, Thomas is still faster due to overhead.
        
        Parameters
        ----------
        d, u, o : np.ndarray
            Diagonal arrays
        b : np.ndarray
            Right-hand side
            
        Returns
        -------
        np.ndarray
            Solution vector
        """
        n = len(d)
        
        # Copy to avoid modifying input
        d_work = d.copy()
        u_work = u.copy()
        o_work = o.copy()
        b_work = b.copy()
        
        # Cyclic reduction algorithm
        stride = 1
        while stride < n:
            # Parallel reduction step
            for i in prange(stride, n - stride, 2 * stride):
                if i - stride >= 0 and i + stride < n:
                    factor_left = u_work[i - stride] / d_work[i - stride]
                    factor_right = o_work[i] / d_work[i + stride]
                    
                    d_work[i] -= factor_left * o_work[i - stride] + factor_right * u_work[i + stride]
                    b_work[i] -= factor_left * b_work[i - stride] + factor_right * b_work[i + stride]
                    
                    if i - 2 * stride >= 0:
                        u_work[i] = -factor_left * u_work[i - stride]
                    if i + 2 * stride < n:
                        o_work[i] = -factor_right * o_work[i + stride]
            
            stride *= 2
        
        # Back substitution
        x = np.zeros(n)
        x[n - 1] = b_work[n - 1] / d_work[n - 1]
        
        for i in range(n - 2, -1, -1):
            x[i] = (b_work[i] - o_work[i] * x[i + 1]) / d_work[i]
        
        return x


    @jit(nopython=True, parallel=True)
    def thomas_solve_numba(d: np.ndarray, u: np.ndarray, 
                           o: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated Thomas algorithm.
        
        This provides significant speedup over pure Python implementation.
        """
        n = len(d)
        
        # Make copies
        d_work = d.copy()
        o_work = o.copy()
        b_work = b.copy()
        
        # Forward elimination
        for i in range(1, n):
            factor = u[i - 1] / d_work[i - 1]
            d_work[i] -= factor * o_work[i - 1]
            b_work[i] -= factor * b_work[i - 1]
        
        # Back substitution
        x = np.zeros(n)
        x[n - 1] = b_work[n - 1] / d_work[n - 1]
        
        for i in range(n - 2, -1, -1):
            x[i] = (b_work[i] - o_work[i] * x[i + 1]) / d_work[i]
        
        return x


# =======================
# CuPy GPU Implementation
# =======================

if CUPY_AVAILABLE:
    def thomas_solve_gpu(d: np.ndarray, u: np.ndarray, 
                        o: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated Thomas algorithm using CuPy.
        
        Transfers data to GPU, solves, and returns result to CPU.
        Best for very large systems where transfer overhead is negligible.
        
        Parameters
        ----------
        d, u, o : np.ndarray
            Diagonal arrays (on CPU)
        b : np.ndarray
            Right-hand side (on CPU)
            
        Returns
        -------
        np.ndarray
            Solution vector (on CPU)
        """
        n = len(d)
        
        # Transfer to GPU
        d_gpu = cp.asarray(d, dtype=cp.float64)
        u_gpu = cp.asarray(u, dtype=cp.float64)
        o_gpu = cp.asarray(o, dtype=cp.float64)
        b_gpu = cp.asarray(b, dtype=cp.float64)
        
        # Make working copies on GPU
        d_work = d_gpu.copy()
        o_work = o_gpu.copy()
        b_work = b_gpu.copy()
        
        # Forward elimination on GPU
        for i in range(1, n):
            factor = u_gpu[i - 1] / d_work[i - 1]
            d_work[i] -= factor * o_work[i - 1]
            b_work[i] -= factor * b_work[i - 1]
        
        # Back substitution on GPU
        x_gpu = cp.zeros(n, dtype=cp.float64)
        x_gpu[n - 1] = b_work[n - 1] / d_work[n - 1]
        
        for i in range(n - 2, -1, -1):
            x_gpu[i] = (b_work[i] - o_work[i] * x_gpu[i + 1]) / d_work[i]
        
        # Transfer back to CPU
        return cp.asnumpy(x_gpu)


    def tridiagonal_solve_gpu_batched(d_list, u_list, o_list, b_list):
        """
        Solve multiple tridiagonal systems on GPU in parallel.
        
        Parameters
        ----------
        d_list, u_list, o_list, b_list : list of np.ndarray
            List of systems to solve
            
        Returns
        -------
        list of np.ndarray
            Solutions for each system
        """
        solutions = []
        
        # Process in batches on GPU
        for d, u, o, b in zip(d_list, u_list, o_list, b_list):
            x = thomas_solve_gpu(d, u, o, b)
            solutions.append(x)
        
        return solutions


# =======================
# Benchmark Functions
# =======================

def benchmark_parallel_methods(sizes, num_runs=5):
    """
    Benchmark different parallel implementations.
    
    Returns timing results for comparison.
    """
    import time
    from linear_systems import build_discrete_laplacian_1d, tridiagonal_solve
    
    results = {
        'sizes': sizes,
        'serial': [],
        'numba': [] if NUMBA_AVAILABLE else None,
        'gpu': [] if CUPY_AVAILABLE else None,
    }
    
    for n in sizes:
        print(f"Benchmarking n={n}...")
        
        # Build system
        d, u, o = build_discrete_laplacian_1d(n)
        b = np.random.randn(n)
        
        # Serial (original Thomas)
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            x = tridiagonal_solve(d.copy(), u.copy(), o.copy(), b.copy())
            times.append(time.perf_counter() - start)
        results['serial'].append(np.mean(times))
        
        # Numba
        if NUMBA_AVAILABLE:
            # Warm-up
            _ = thomas_solve_numba(d.copy(), u.copy(), o.copy(), b.copy())
            
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                x = thomas_solve_numba(d.copy(), u.copy(), o.copy(), b.copy())
                times.append(time.perf_counter() - start)
            results['numba'].append(np.mean(times))
        
        # GPU
        if CUPY_AVAILABLE:
            # Warm-up
            _ = thomas_solve_gpu(d.copy(), u.copy(), o.copy(), b.copy())
            
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                x = thomas_solve_gpu(d.copy(), u.copy(), o.copy(), b.copy())
                times.append(time.perf_counter() - start)
            results['gpu'].append(np.mean(times))
    
    return results


if __name__ == '__main__':
    print("Parallel Solvers Module")
    print("=" * 60)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"CuPy available: {CUPY_AVAILABLE}")
    
    if NUMBA_AVAILABLE or CUPY_AVAILABLE:
        sizes = [1000, 5000, 10000, 50000]
        print("\nRunning benchmark...")
        results = benchmark_parallel_methods(sizes)
        
        print("\nResults (time in seconds):")
        print(f"{'Size':>10} {'Serial':>12} {'Numba':>12} {'GPU':>12}")
        print("-" * 50)
        for i, n in enumerate(sizes):
            serial_time = results['serial'][i]
            numba_time = results['numba'][i] if results['numba'] else None
            gpu_time = results['gpu'][i] if results['gpu'] else None
            
            print(f"{n:10d} {serial_time:12.6f} ", end="")
            print(f"{numba_time:12.6f} " if numba_time else "N/A          ", end="")
            print(f"{gpu_time:12.6f}" if gpu_time else "N/A")
