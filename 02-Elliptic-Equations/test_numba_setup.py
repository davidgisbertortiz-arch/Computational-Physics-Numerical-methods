#!/usr/bin/env python3
"""
Quick test script to verify Numba installation and performance.

Usage:
    python test_numba_setup.py
"""

import sys
import time
import numpy as np

def test_numba_availability():
    """Check if Numba is installed and working."""
    print("="*70)
    print("Numba Installation Check")
    print("="*70)
    
    try:
        import numba
        print(f"✅ Numba version: {numba.__version__}")
        print(f"✅ Threading layer: {numba.config.THREADING_LAYER}")
        
        # Check OpenMP
        import os
        n_threads = os.environ.get('OMP_NUM_THREADS', 'auto')
        print(f"✅ OMP_NUM_THREADS: {n_threads}")
        
        return True
    except ImportError:
        print("❌ Numba not installed")
        print("   Install with: pip install numba")
        return False

def test_jit_compilation():
    """Test basic JIT compilation."""
    print("\n" + "="*70)
    print("JIT Compilation Test")
    print("="*70)
    
    try:
        from numba import njit
        
        @njit
        def sum_array(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        # Test array
        arr = np.random.randn(10000)
        
        # First call (includes compilation time)
        t0 = time.time()
        result1 = sum_array(arr)
        t1 = time.time() - t0
        
        # Second call (compiled)
        t0 = time.time()
        result2 = sum_array(arr)
        t2 = time.time() - t0
        
        # NumPy for comparison
        t0 = time.time()
        result3 = np.sum(arr)
        t3 = time.time() - t0
        
        print(f"✅ JIT compilation successful")
        print(f"   First call (with compilation): {t1*1000:.3f} ms")
        print(f"   Second call (compiled):        {t2*1000:.6f} ms")
        print(f"   NumPy (reference):             {t3*1000:.6f} ms")
        print(f"   Speedup: {t3/t2:.1f}× faster than NumPy")
        
        # Verify results match
        assert np.isclose(result1, result3), "Results don't match!"
        assert np.isclose(result2, result3), "Results don't match!"
        
        return True
    except Exception as e:
        print(f"❌ JIT compilation failed: {e}")
        return False

def test_parallel_execution():
    """Test parallel execution with prange."""
    print("\n" + "="*70)
    print("Parallel Execution Test")
    print("="*70)
    
    try:
        from numba import njit, prange
        
        @njit(parallel=False)
        def sum_2d_sequential(arr):
            total = 0.0
            n, m = arr.shape
            for i in range(n):
                for j in range(m):
                    total += arr[i, j]
            return total
        
        @njit(parallel=True)
        def sum_2d_parallel(arr):
            total = 0.0
            n, m = arr.shape
            for i in prange(n):
                for j in range(m):
                    total += arr[i, j]
            return total
        
        # Large array
        arr = np.random.randn(1000, 1000)
        
        # Warm-up (compile)
        _ = sum_2d_sequential(arr)
        _ = sum_2d_parallel(arr)
        
        # Sequential
        t0 = time.time()
        result_seq = sum_2d_sequential(arr)
        t_seq = time.time() - t0
        
        # Parallel
        t0 = time.time()
        result_par = sum_2d_parallel(arr)
        t_par = time.time() - t0
        
        print(f"✅ Parallel execution successful")
        print(f"   Sequential time: {t_seq*1000:.3f} ms")
        print(f"   Parallel time:   {t_par*1000:.3f} ms")
        print(f"   Speedup:         {t_seq/t_par:.2f}×")
        
        # Verify results match
        assert np.isclose(result_seq, result_par, rtol=1e-6), "Results don't match!"
        
        if t_seq/t_par < 1.2:
            print("   ⚠️  Warning: Speedup less than 1.2× (expected 2-4×)")
            print("      Try setting OMP_NUM_THREADS environment variable")
        
        return True
    except Exception as e:
        print(f"❌ Parallel execution failed: {e}")
        return False

def test_2d_stencil():
    """Test 5-point stencil operation (relevant to elliptic solver)."""
    print("\n" + "="*70)
    print("2D Stencil Operation Test")
    print("="*70)
    
    try:
        from numba import njit, prange
        
        @njit(parallel=True, fastmath=True)
        def apply_laplacian(u, u_new, h):
            """Apply 5-point Laplacian stencil."""
            n, m = u.shape
            h2_inv = 1.0 / (h * h)
            
            for i in prange(1, n-1):
                for j in range(1, m-1):
                    u_new[i, j] = h2_inv * (
                        u[i-1, j] + u[i+1, j] + 
                        u[i, j-1] + u[i, j+1] - 
                        4.0 * u[i, j]
                    )
        
        # Test grid
        n = 500
        u = np.random.randn(n, n)
        u_new = np.zeros((n, n))
        h = 1.0 / (n - 1)
        
        # Warm-up
        apply_laplacian(u, u_new, h)
        
        # Benchmark
        n_runs = 10
        t0 = time.time()
        for _ in range(n_runs):
            apply_laplacian(u, u_new, h)
        t_total = (time.time() - t0) / n_runs
        
        # Compute throughput
        n_ops = (n-2)**2 * 7  # 7 ops per interior point
        throughput = n_ops / t_total / 1e9  # GFLOPS
        
        print(f"✅ Stencil operation successful")
        print(f"   Grid size:   {n}×{n}")
        print(f"   Time:        {t_total*1000:.3f} ms")
        print(f"   Throughput:  {throughput:.2f} GFLOPS")
        
        return True
    except Exception as e:
        print(f"❌ Stencil operation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "NUMBA SETUP TEST" + " "*32 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    results = []
    
    # Test 1: Installation
    results.append(("Installation", test_numba_availability()))
    
    if not results[0][1]:
        print("\n❌ Numba not available. Install with: pip install numba")
        sys.exit(1)
    
    # Test 2: JIT compilation
    results.append(("JIT Compilation", test_jit_compilation()))
    
    # Test 3: Parallel execution
    results.append(("Parallel Execution", test_parallel_execution()))
    
    # Test 4: Stencil operation
    results.append(("2D Stencil", test_2d_stencil()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<25} {status}")
    
    print("="*70)
    
    if all_passed:
        print("\n✅ All tests passed! Numba is ready for use.")
        print("   You can now run notebook 08_performance_optimization.ipynb")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
        print("   Notebook Part 5 (parallel solvers) may not work properly.")
    
    print()

if __name__ == "__main__":
    main()
