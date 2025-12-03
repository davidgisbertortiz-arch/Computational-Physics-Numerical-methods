# Chapter 02 Report

This directory contains the comprehensive LaTeX report for Chapter 02: Elliptic Equations.

## Contents

- `chapter02_elliptic_equations.tex`: Main LaTeX source file
- `Makefile`: Automated compilation

## Quick Compilation

```bash
# Using Makefile (recommended)
make

# Or manually with pdflatex
pdflatex chapter02_elliptic_equations.tex
pdflatex chapter02_elliptic_equations.tex  # Second pass for references
```

## Report Structure

1. **Introduction**
   - 2D Poisson equation definition
   - Boundary conditions (Dirichlet, Neumann)
   - Discretization approach

2. **Discrete Laplacian Operator**
   - 5-point stencil derivation
   - Kronecker product structure
   - Sparse matrix representation

3. **Solver Implementations**
   - Direct solver (sparse LU)
   - Conjugate Gradient (CG)
   - Point-iterative methods (Jacobi, SOR)
   - Line-relaxation methods
   - Alternating Direction Implicit (ADI)

4. **Tensor Formulation**
   - 4D tensor representation
   - Tensor contraction interpretation
   - Why tensors are impractical for 2D
   - When tensor approaches are useful

5. **Numerical Experiments**
   - Test problem definition
   - Performance benchmarks
   - Convergence analysis
   - Scaling studies

6. **Implementation Details**
   - Integration with Chapter 01
   - Boundary condition handling
   - Bug fix documentation

7. **Conclusions**
   - Solver recommendations
   - Summary of contributions
   - Future work

## Figures

The report references figures from the notebooks:
- `../notebooks/figures/convergence_comparison.png`
- `../notebooks/figures/timing_comparison.png`
- Additional plots from notebook 04

Make sure to run the notebooks before compiling to generate all figures.

## Requirements

**LaTeX packages used**:
- `amsmath`, `amssymb`, `amsthm` (mathematics)
- `graphicx`, `subcaption` (figures)
- `listings` (code blocks)
- `algorithm`, `algpseudocode` (algorithms)
- `booktabs` (professional tables)
- `hyperref` (cross-references and links)

Most LaTeX distributions (TeX Live, MiKTeX) include these by default.

## Makefile Targets

```bash
make          # Compile report (default)
make clean    # Remove auxiliary files
make view     # Open PDF in system viewer
make all      # Clean and compile
make help     # Show available targets
```

## Manual Compilation

If you don't have `make`:

```bash
# Compile twice for proper references
pdflatex chapter02_elliptic_equations.tex
pdflatex chapter02_elliptic_equations.tex

# View the result
open chapter02_elliptic_equations.pdf  # macOS
xdg-open chapter02_elliptic_equations.pdf  # Linux
```

## Output

The compiled report is `chapter02_elliptic_equations.pdf`.

## License

Same as the main repository (see LICENSE in repository root).
