# Chapter 03: Eigenvalue Problems

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6?logo=scipy&logoColor=white" alt="SciPy"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/LaTeX-Report-008080?logo=latex&logoColor=white" alt="LaTeX"/>
</p>

This chapter provides a comprehensive implementation of **iterative eigenvalue methods** commonly used in computational physics, from basic power iteration to the Lanczos algorithm for large sparse systems.

## 🎯 The Eigenvalue Problem

Find scalar λ and non-zero vector **v** such that:

$$A\mathbf{v} = \lambda \mathbf{v}$$

This fundamental problem appears throughout physics:
- **Quantum mechanics**: Schrödinger equation eigenvalues are energy levels
- **Vibrations**: Normal modes and natural frequencies
- **Stability analysis**: System stability from eigenvalue signs
- **Data science**: PCA, spectral clustering, PageRank

## 📚 Implemented Methods

| Method | Convergence | Finds | Best For |
|--------|-------------|-------|----------|
| **Power Iteration** | Linear: O(\|λ₂/λ₁\|ᵏ) | Dominant eigenvalue | PageRank, simple matrices |
| **Inverse Iteration** | Linear: O(\|λᵢ-σ\|/\|λⱼ-σ\|)ᵏ | Eigenvalue nearest to σ | Refinement, specific modes |
| **Rayleigh Quotient** | **Cubic**: O(ε³ᵏ) | Single eigenvalue | High-precision applications |
| **QR Iteration** | Quadratic (with shifts) | All eigenvalues | Small-to-medium dense matrices |
| **Lanczos** | Depends on spectrum | Extremal eigenvalues | Large sparse symmetric matrices |

### Convergence Comparison

```
Error vs Iterations (log scale)
    │
10⁰ ┤ ●───────────────────────────────────── Power (linear)
    │   ●
10⁻³┤     ■─────────────────────────────── Inverse (linear, faster)
    │       ■
10⁻⁶┤         ▲───────────────────────── Rayleigh (cubic!)
    │           ▲
10⁻⁹┤             ▲
    │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┴
        0   2   4   6   8  10  12  14  16
                    Iterations
```

## 🏗️ Project Structure

```
03-Eigenvalue-Problems/
├── src/
│   └── eigensolvers.py          # Core implementations (~1100 lines)
│       ├── power_iteration()     # Dominant eigenvalue
│       ├── inverse_iteration()   # Shifted inverse iteration
│       ├── rayleigh_quotient_iteration()  # Cubic convergence
│       ├── qr_iteration_basic()  # All eigenvalues (educational)
│       ├── lanczos()            # Large sparse symmetric
│       └── verify_eigenpair()   # Verification utility
├── tests/
│   ├── test_eigensolvers.py     # Core method tests
│   ├── test_lanczos.py          # Lanczos-specific tests
│   └── verify_eigensolvers.ipynb # Interactive verification
├── notebooks/
│   └── 01_eigenvalue_methods.ipynb  # Complete tutorial with visualizations
├── report/
│   ├── chapter03_eigenvalue_problems.tex  # Full mathematical theory
│   └── chapter03_eigenvalue_problems.pdf  # Compiled report
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
cd 03-Eigenvalue-Problems
pip install -r requirements.txt
```

### Basic Usage

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

from eigensolvers import (
    power_iteration,
    inverse_iteration, 
    rayleigh_quotient_iteration,
    qr_iteration_basic,
    lanczos
)
import numpy as np

# Symmetric test matrix
A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 2]], dtype=float)

# 1. Dominant eigenvalue (power iteration)
lam_max, v_max, info = power_iteration(A, tol=1e-10)
print(f"Dominant eigenvalue: {lam_max:.10f}")
print(f"Iterations: {info['iterations']}")

# 2. Eigenvalue closest to σ=2.5 (inverse iteration)
lam_near, v_near, info = inverse_iteration(A, sigma=2.5, tol=1e-10)
print(f"Eigenvalue near 2.5: {lam_near:.10f}")

# 3. High-precision refinement (Rayleigh quotient)
lam_rq, v_rq, info = rayleigh_quotient_iteration(A, sigma0=4.5, tol=1e-14)
print(f"Rayleigh result: {lam_rq:.14f} in {info['iterations']} iterations")

# 4. All eigenvalues (QR iteration)
eigenvalues, Q, info = qr_iteration_basic(A)
print(f"All eigenvalues: {np.sort(eigenvalues)}")
```

### Working with Large Sparse Matrices

```python
from scipy.sparse import diags

# 1D Laplacian (tridiagonal, sparse)
n = 1000
h = 1.0 / (n + 1)
diagonals = [[-1]*n, [2]*n, [-1]*n]
L = diags(diagonals, [-1, 0, 1], format='csr') / h**2

# Find smallest eigenvalues with Lanczos
eigenvalues, eigenvectors, info = lanczos(L, k=5, which='smallest')
print(f"5 smallest eigenvalues: {eigenvalues}")
```

## 📊 Physical Applications

### Normal Modes of Coupled Oscillators

```python
# Three masses connected by springs
# M * x'' = -K * x  →  (M^-1 K) v = ω² v

M = np.diag([1.0, 1.0, 1.0])  # Mass matrix
K = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])     # Stiffness matrix

# Solve generalized eigenvalue problem
A = np.linalg.solve(M, K)
omega_squared, modes, _ = qr_iteration_basic(A)
frequencies = np.sqrt(np.abs(omega_squared))
print(f"Natural frequencies ω: {frequencies}")
```

### Quantum Harmonic Oscillator

The Lanczos method is ideal for finding ground state and low-lying excited states of large Hamiltonian matrices in quantum mechanics.

## 🧪 Running Tests

```bash
# Run all tests
cd 03-Eigenvalue-Problems
pytest tests/ -v

# Run specific test file
pytest tests/test_eigensolvers.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📖 Mathematical Theory

The `report/` directory contains a comprehensive LaTeX document covering:

1. **Problem Formulation**: Definitions, spectral theorem
2. **Power Iteration**: Convergence analysis, Rayleigh quotient
3. **Inverse Iteration**: Shift-and-invert strategy
4. **Rayleigh Quotient Iteration**: Cubic convergence proof
5. **QR Algorithm**: Similarity transformations, deflation
6. **Lanczos Method**: Krylov subspaces, Ritz values
7. **Numerical Stability**: Orthogonalization, reorthogonalization
8. **Physical Applications**: Vibrations, quantum mechanics

To compile the PDF:
```bash
cd report
pdflatex chapter03_eigenvalue_problems.tex
pdflatex chapter03_eigenvalue_problems.tex  # Run twice for TOC
```

## 🔗 Dependencies

### Required
- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7`

### Optional
- `numba`: JIT compilation for performance (~10x speedup)
- `jupyter`: Interactive notebooks

### Inter-Chapter Dependencies

This chapter can use utilities from **Chapter 01: Linear Systems**:
- `tridiagonal_solve`: Thomas algorithm for inverse iteration
- `spectral_radius`: For convergence analysis

If Chapter 01 is unavailable, NumPy/SciPy fallbacks are automatically used.

## 📈 Performance Notes

| Matrix Size | Power | Inverse | Rayleigh | QR | Lanczos (k=5) |
|-------------|-------|---------|----------|-----|---------------|
| 10×10 | 0.1 ms | 0.1 ms | 0.1 ms | 0.5 ms | N/A |
| 100×100 | 0.3 ms | 0.5 ms | 0.2 ms | 15 ms | 2 ms |
| 1000×1000 | 5 ms | 8 ms | 3 ms | 2 s | 20 ms |
| 10000×10000 | 80 ms | 150 ms | 50 ms | N/A | 200 ms |

*Times measured with NumPy on Intel i7. Lanczos assumes sparse matrix.*

## 📝 API Reference

### `power_iteration(A, v0=None, tol=1e-10, max_iter=1000, return_history=False)`
Returns: `(eigenvalue, eigenvector, info_dict)`

### `inverse_iteration(A, sigma=0.0, v0=None, tol=1e-10, max_iter=100, return_history=False)`
Returns: `(eigenvalue, eigenvector, info_dict)`

### `rayleigh_quotient_iteration(A, v0=None, sigma0=None, tol=1e-10, max_iter=50, return_history=False)`
Returns: `(eigenvalue, eigenvector, info_dict)`

### `qr_iteration_basic(A, max_iter=1000, tol=1e-10, return_history=False)`
Returns: `(eigenvalues_array, Q_matrix, info_dict)`

### `lanczos(A, k=6, v0=None, which='largest', tol=1e-10, max_iter=None)`
Returns: `(eigenvalues, eigenvectors, info_dict)`

## 📚 References

1. Trefethen, L.N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
2. Golub, G.H. & Van Loan, C.F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins.
3. Parlett, B.N. (1998). *The Symmetric Eigenvalue Problem*. SIAM.
4. Saad, Y. (2011). *Numerical Methods for Large Eigenvalue Problems* (2nd ed.). SIAM.

---

<p align="center">
  <b>Part of the Computational Physics: Numerical Methods series</b><br>
  <a href="../01-Linear-Systems">← Chapter 01: Linear Systems</a> |
  <a href="../02-Elliptic-Equations">Chapter 02: Elliptic Equations</a>
</p>
