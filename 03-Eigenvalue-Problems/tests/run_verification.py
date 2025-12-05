#!/usr/bin/env python3
"""
Script de verificación de eigensolvers.
Ejecutar: python run_verification.py
"""
import sys
from pathlib import Path

# Setup paths
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root.parent / '01-Linear-Systems' / 'src'))

import numpy as np

# Import eigensolvers
from eigensolvers import (
    power_iteration, inverse_iteration, rayleigh_quotient_iteration,
    qr_iteration_basic, rayleigh_quotient, verify_eigenpair,
    find_dominant_eigenvalue, find_smallest_eigenvalue, find_eigenvalue_near,
    SCIPY_AVAILABLE, CHAPTER01_AVAILABLE
)

def test_power_iteration():
    """Test power iteration."""
    A = np.array([[4, 1], [1, 3]], dtype=float)
    eigenvalue, eigenvector, iterations = power_iteration(A)
    expected = max(np.linalg.eigvalsh(A))
    error = abs(eigenvalue - expected)
    assert error < 1e-6, f"Power iteration failed: error={error}"
    return True

def test_inverse_iteration():
    """Test inverse iteration."""
    A = np.array([[4, 1], [1, 3]], dtype=float)
    sigma = 2.5
    eigenvalue, eigenvector, iterations = inverse_iteration(A, sigma=sigma)
    expected = min(np.linalg.eigvalsh(A))
    error = abs(eigenvalue - expected)
    assert error < 1e-6, f"Inverse iteration failed: error={error}"
    return True

def test_rayleigh_quotient_iteration():
    """Test Rayleigh quotient iteration."""
    A = np.array([[4, 1], [1, 3]], dtype=float)
    v0 = np.array([1, 0], dtype=float)
    sigma0 = 4.5
    eigenvalue, eigenvector, iterations = rayleigh_quotient_iteration(A, v0=v0, sigma0=sigma0)
    expected = max(np.linalg.eigvalsh(A))
    error = abs(eigenvalue - expected)
    assert error < 1e-6, f"Rayleigh quotient iteration failed: error={error}"
    return True

def test_qr_iteration():
    """Test QR iteration."""
    A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
    eigenvalues, iterations = qr_iteration_basic(A)
    expected = np.sort(np.linalg.eigvalsh(A))
    error = np.max(np.abs(np.sort(eigenvalues) - expected))
    assert error < 1e-6, f"QR iteration failed: max error={error}"
    return True

def test_convenience_functions():
    """Test convenience functions."""
    A = np.array([[5, 1, 0], [1, 4, 1], [0, 1, 3]], dtype=float)
    eigvals_np = np.linalg.eigvalsh(A)
    
    lam_dom = find_dominant_eigenvalue(A)
    assert abs(lam_dom - max(eigvals_np)) < 1e-6, "find_dominant_eigenvalue failed"
    
    lam_small = find_smallest_eigenvalue(A)
    assert abs(lam_small - min(eigvals_np)) < 1e-6, "find_smallest_eigenvalue failed"
    
    target = 4.0
    lam_near = find_eigenvalue_near(A, target)
    closest = eigvals_np[np.argmin(np.abs(eigvals_np - target))]
    assert abs(lam_near - closest) < 1e-6, "find_eigenvalue_near failed"
    
    return True

def test_large_matrix():
    """Test with larger matrix."""
    np.random.seed(42)
    n = 10
    B = np.random.randn(n, n)
    A = B @ B.T  # Symmetric positive definite
    
    lam_power, _, _ = power_iteration(A, tol=1e-10)
    expected_max = max(np.linalg.eigvalsh(A))
    assert abs(lam_power - expected_max) < 1e-6, f"Large matrix power iteration failed"
    
    return True

def main():
    print("=" * 60)
    print("VERIFICACIÓN DE EIGENSOLVERS - Chapter 03")
    print("=" * 60)
    print()
    print(f"SCIPY_AVAILABLE: {SCIPY_AVAILABLE}")
    print(f"CHAPTER01_AVAILABLE: {CHAPTER01_AVAILABLE}")
    print()
    
    tests = [
        ("Power Iteration", test_power_iteration),
        ("Inverse Iteration", test_inverse_iteration),
        ("Rayleigh Quotient Iteration", test_rayleigh_quotient_iteration),
        ("QR Iteration", test_qr_iteration),
        ("Convenience Functions", test_convenience_functions),
        ("Large Matrix (10x10)", test_large_matrix),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTADOS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE 🎉\n")
        return 0
    else:
        print("\n❌ ALGUNOS TESTS FALLARON ❌\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
