# Chapter 02: Elliptic Equations (Poisson / Laplace)

This chapter contains utilities and examples for solving 2D elliptic equations (Laplace/Poisson) on rectangular domains, with constant Dirichlet and Neumann boundary conditions. It is designed to be pedagogical and efficient (using sparse matrices and iterative/preconditioned methods).

**INTEGRATION WITH CHAPTER 01:** This chapter leverages advanced utilities from Chapter 01 (tridiagonal linear systems) to implement high-performance solvers: line-relaxation, ADI, and optional support for Numba/CuPy acceleration.

## Repository Structure

- `src/elliptic.py`: Functions to build the discrete operator (Kronecker form), boundary condition handling, and solvers:
  - **Direct solver** (sparse LU via `spsolve`)
  - **CG** (Conjugate Gradient with preconditioning)
  - **Point iterative methods**: Jacobi, SOR
  - **Line-relaxation (line-SOR)**: uses `tridiagonal_solve` from Chapter 01 to solve lines implicitly
  - **ADI (Alternating Direction Implicit)**: uses Thomas algorithm for fast implicit solves in alternating directions
  - **Batched tridiagonal solver wrapper**: optional Numba/CuPy acceleration for batched line solves
- `run_elliptic.py`: Demo script that solves examples and generates figures
- `notebooks/`:
  - `01_elliptic_intro.ipynb`: Intro notebook with basic solvers
  - `02_runnable_tutorial.ipynb`: Timing benchmarks, saves CSV and figures
  - `03_neumann_and_preconditioning.ipynb`: Neumann BC, Poisson sources, ILU+GMRES, optional PyAMG multigrid
  - `04_advanced_analysis.ipynb`: **COMPREHENSIVE ANALYSIS** with convergence plots, speedup analysis, 3D surfaces, heatmaps, iteration comparisons
  - `05_tensor_formulation.ipynb`: **TENSOR APPROACH** - Novel 4D tensor formulation of the Laplacian (educational, shows why sparse is optimal)
- `report/`: LaTeX report with mathematical background and performance analysis
- `requirements.txt`: Minimal dependencies for this chapter (numpy, scipy, matplotlib, pandas, optional: pyamg)
- `tests/test_elliptic.py`: Basic unit tests
- `tests/test_advanced_solvers.py`: Tests for line-relaxation, ADI using Chapter 01 utilities

## Advanced Features from Chapter 01 Integration

### Line-Relaxation (Line-SOR)
Instead of updating grid points individually (point Jacobi/SOR), line-relaxation solves an entire line of unknowns simultaneously using the Thomas algorithm. This dramatically reduces iteration counts and improves convergence.

**Key advantage:** Each line solve is O(n) using `tridiagonal_solve`, much faster than matrix inversion.

### ADI Method
The Alternating Direction Implicit method splits the 2D problem into sequences of 1D implicit solves. Each ADI half-step solves tridiagonal systems using the Thomas algorithm, providing unconditional stability and fast convergence.

### Batched Solvers with Numba/CuPy
The `batched_thomas` function wraps Chapter 01's parallel solvers (`thomas_solve_numba`, `thomas_solve_gpu`) to accelerate multiple tridiagonal solves—useful for multigrid smoothers or ADI on large grids.

## Quick Start

```bash
# Install dependencies
pip install -r 02-Elliptic-Equations/requirements.txt

# Run demo
python 02-Elliptic-Equations/run_elliptic.py

# Run tests
pytest -v 02-Elliptic-Equations/tests

# Run notebooks (from repository root)
cd 02-Elliptic-Equations && jupyter lab notebooks/
```

## Compile LaTeX Report

```bash
cd 02-Elliptic-Equations/report
pdflatex chapter02_elliptic_equations.tex
pdflatex chapter02_elliptic_equations.tex  # Second pass for references
```

## Key Results

**Performance on 80×60 grid (3364 unknowns)**:
- **Fastest**: CG (0.007s, 58 iters, 2.1×10⁻⁵ error)
- **Most accurate**: ADI (16.8s, 1503 iters, 2.5×10⁻⁸ error)
- **Best balance**: Line-SOR (17.9s, 3092 iters, 5.4×10⁻⁸ error)

**Tensor formulation**: Educational value only - 500× slower and 500× more memory than sparse matrices for n=30.

## Solver Recommendations

| Use Case | Recommended Solver | Rationale |
|----------|-------------------|-----------|
| **Rapid prototyping** | Direct sparse LU | Simple, exact solution |
| **Best performance** | CG with ILU | Fastest for moderate accuracy |
| **Extreme accuracy** | ADI or Line-SOR | 10⁻⁸ relative error achievable |
| **GPU acceleration** | CG or Line-SOR | Naturally parallel algorithms |
| **Teaching/education** | Tensor formulation | Intuitive 2D structure |

## Documentation

- `README.md`: Quick overview (Spanish)
- `README_EN.md` (this file): Quick overview (English)
- `STATUS.md`: Implementation status and test results
- `CHAPTER_INVENTORY.md`: Complete feature inventory
- `BUG_FIX_SUMMARY.md`: Technical details of the operator splitting bug fix
- `NOTEBOOK_CLEANUP_SUMMARY.md`: Documentation of notebook 04 improvements
- `report/chapter02_elliptic_equations.tex`: Comprehensive mathematical report with performance analysis
- `notebooks/README.md`: Notebook-specific documentation and learning paths

## Testing

All solvers have comprehensive unit tests:

```bash
# Run all tests
pytest 02-Elliptic-Equations/tests/ -v

# Run only basic tests
pytest 02-Elliptic-Equations/tests/test_elliptic.py -v

# Run only advanced solver tests
pytest 02-Elliptic-Equations/tests/test_advanced_solvers.py -v
```

## Notebooks Guide

1. **01_elliptic_intro.ipynb**: Start here for basic understanding
   - Direct solver introduction
   - Simple boundary conditions
   - 2D visualization

2. **02_runnable_tutorial.ipynb**: Timing and performance benchmarks
   - Automated benchmark generation
   - CSV export of results
   - Comparative analysis

3. **03_neumann_and_preconditioning.ipynb**: Advanced boundary conditions
   - Neumann BC implementation
   - Poisson equation with source terms
   - ILU preconditioning
   - GMRES solver
   - Optional multigrid (PyAMG)

4. **04_advanced_analysis.ipynb**: Publication-quality analysis
   - Convergence rate plots
   - 3D surface visualizations
   - Speedup analysis
   - Trade-off comparisons
   - Executive summary

5. **05_tensor_formulation.ipynb**: Novel mathematical perspective
   - 4D tensor representation of Laplacian
   - Tensor vs sparse comparison
   - Scaling analysis
   - Iterative tensor solvers
   - Kronecker decomposition
   - **Educational only** - demonstrates why sparse is optimal

## Bug Fix History

**Major bug fixed (December 2024)**: Line-relaxation and ADI methods initially used incorrect diagonal when splitting the 2D operator. The bug caused NaN/divergence issues.

**Root cause**: Used only x-direction contribution (`-2/hx²`) instead of full 2D diagonal (`-2/hx² - 2/hy²`).

**Solution**: Modified diagonal before tridiagonal solve:
```python
d_modified = d_x - 2.0/hy²
```

**Result**: Errors dropped from O(1) to machine precision (10⁻¹⁵). See `BUG_FIX_SUMMARY.md` for complete technical details.

## Requirements

**Minimal** (required):
- `numpy >= 1.20`
- `scipy >= 1.7`
- `matplotlib >= 3.5`
- `pandas >= 1.3` (for benchmarking)
- `pytest >= 6.0` (for testing)

**Optional** (for acceleration):
- `numba >= 0.55` (JIT compilation, parallel CPU)
- `cupy >= 10.0` (GPU acceleration via CUDA)
- `pyamg >= 4.2` (algebraic multigrid)

## Future Extensions

Potential additions for future development:
1. **Multigrid methods**: O(N) complexity for arbitrary accuracy
2. **GPU implementations**: CUDA kernels for line-based methods
3. **Adaptive mesh refinement**: Local grid refinement for complex geometries
4. **3D elliptic problems**: Extension to ∇²u in 3D
5. **Tensor networks**: Low-rank approximations for high-dimensional PDEs
6. **Neural operators**: Physics-informed neural networks (PINNs)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{computational-physics-methods,
  author = {Your Name},
  title = {Computational Physics: Numerical Methods - Chapter 02: Elliptic Equations},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/davidgisbertortiz-arch/Computational-Physics-Numerical-methods}
}
```

## License

See LICENSE file in repository root.
