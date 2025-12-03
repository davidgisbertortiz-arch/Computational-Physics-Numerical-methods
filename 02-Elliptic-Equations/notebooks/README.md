# Elliptic Equations: Jupyter Notebooks

This directory contains interactive Jupyter notebooks for exploring numerical methods for elliptic PDEs (2D Poisson/Laplace equations).

## Notebook Overview

### 01_introduction.ipynb
**Beginner-friendly introduction**
- Basic concepts of elliptic PDEs
- Finite difference discretization
- Simple 1D examples
- Visualization of solution structure

**Prerequisites**: None (start here!)

---

### 02_runnable_tutorial.ipynb
**Hands-on tutorial with working examples**
- Setting up and solving 2D Poisson equation
- Comparison of Direct vs CG solvers
- Introduction to Line-SOR and ADI methods
- Quick benchmarking (small grids)

**Prerequisites**: 01_introduction.ipynb

**Key Learning**: How to use the `elliptic.py` module for practical problems

---

### 03_neumann_and_preconditioning.ipynb
**Advanced boundary conditions and preconditioning**
- Neumann boundary conditions (flux/gradient BCs)
- ILU preconditioning for GMRES
- Comparison with ADI on mixed BCs
- Practical examples (heat flux problems)

**Prerequisites**: 02_runnable_tutorial.ipynb

**Key Learning**: Handling realistic boundary conditions

---

### 04_advanced_analysis.ipynb ⭐
**Comprehensive performance benchmarking and analysis**
- In-depth comparison of 6 solvers:
  - Direct (LU factorization)
  - CG (with Jacobi preconditioner)
  - Jacobi, SOR (point methods)
  - Line-SOR, ADI (line methods)

**Prerequisites**: 02_runnable_tutorial.ipynb

**Key Features**:
- Publication-quality visualizations (300 DPI)
- Convergence rate analysis
- 3D surface plots
- Trade-off comparisons (speed vs accuracy)
- Executive summary and recommendations

---

### 05_tensor_formulation.ipynb 🎓
**Novel 4D tensor approach to elliptic PDEs**
- Mathematical theory of 4D Laplacian tensor
- Tensor contraction interpretation
- Direct tensor solver using `np.linalg.tensorsolve`
- Iterative tensor methods (Jacobi, Gauss-Seidel)
- Comprehensive comparison: Tensor vs Sparse
- Scaling analysis (memory, time, accuracy)
- Kronecker product decomposition
- **Why optimized tensor libraries don't help**

**Prerequisites**: 04_advanced_analysis.ipynb (for comparison context)

**Key Learning**: 
- Understanding operator structure at a fundamental level
- Why sparse matrices are optimal for 2D problems
- When tensor approaches are useful (high dimensions, ML integration)
- Educational value vs practical performance

**Educational Value**: ⭐⭐⭐⭐⭐ (excellent for understanding)
**Practical Value**: ⭐ (only for education - not for production)
- Performance metrics: time, iterations, accuracy
- 7 visualizations:
  1. Timing comparison bar chart
  2. Iteration count comparison
  3. Solution heatmaps (temperature distribution)
  4. 3D surface plot
  5. Convergence history (residual decay)
  6. Speedup analysis
  7. Accuracy vs speed trade-off
- Detailed conclusions and practical recommendations

**Prerequisites**: 02_runnable_tutorial.ipynb (recommended: 03 as well)

**Key Learning**: 
- **ADI achieves 100-1000× better accuracy** than point methods
- Line-based methods are optimal for 2D elliptic problems
- Trade-offs between speed and accuracy
- Production-ready solver selection

**Grid Size**: 80×60 (4,680 unknowns) - representative of real problems

---

### 05_supercomputer_techniques.ipynb
**High-performance computing methods** (optional/advanced)
- Geometric multigrid
- Parallel implementations
- GPU acceleration with CuPy
- Scaling to large grids (>1M unknowns)

**Prerequisites**: 04_advanced_analysis.ipynb

**Note**: Requires optional dependencies (Numba, CuPy)

---

### 06_physics_applications.ipynb
**Real-world applications** (optional/applied)
- Electrostatics (Poisson equation for potential)
- Heat conduction in complex geometries
- Fluid flow (stream function formulation)
- Comparison of solvers on practical problems

**Prerequisites**: 03 and 04

---

## Quick Start

### Setup
```bash
# From repository root
cd 02-Elliptic-Equations

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

### Recommended Path
1. **Learning path** (for students/beginners):
   - 01 → 02 → 03 → 04 → (optional: 06)

2. **Research/production path** (for practitioners):
   - 02 → 04 → (skim 03 if needed) → 05 (if HPC required)

3. **Quick benchmark** (for solver selection):
   - Jump directly to **04_advanced_analysis.ipynb**
   - Check Executive Summary section
   - Look at accuracy vs speed trade-off plot

---

## Key Results (from Notebook 04)

| Solver | Time (s) | Rel. Error | Best Use Case |
|--------|----------|------------|---------------|
| **CG** | 0.007 | 2.1×10⁻⁵ | Rapid prototyping |
| Direct | 0.014 | - | Reference/small grids |
| Jacobi | 0.116 | 1.8×10⁻² | Educational only |
| **ADI** | 16.8 | 2.5×10⁻⁸ | **Production (best accuracy)** |
| Line-SOR | 17.9 | 5.4×10⁻⁸ | Production (irregular domains) |
| SOR | 29.0 | 1.5×10⁻⁶ | Avoid for 2D |

**Recommendation**: Use **ADI** or **Line-SOR** for production 2D elliptic problems. They achieve exceptional accuracy (10⁻⁸ level) with reasonable computational cost.

---

## Troubleshooting

### Import Errors
Make sure you're running from the repository root and the paths are set correctly:
```python
import sys
from pathlib import Path
repo_root = Path.cwd().parent.parent  # Adjust based on notebook location
sys.path.insert(0, str(repo_root / '02-Elliptic-Equations' / 'src'))
sys.path.insert(0, str(repo_root / '01-Linear-Systems' / 'src'))
```

### Convergence Issues
- **ADI/Line-SOR not converging?** Increase `maxiter`:
  ```python
  x, it, res = adi_solve(..., maxiter=2000)  # Default was 500
  ```
- **Getting NaN?** Check boundary conditions are properly specified
- **Slow convergence?** Try different relaxation parameter `omega` (1.0-1.5)

### Performance Issues
- Notebook running slow? Reduce grid size:
  ```python
  nx, ny = 40, 30  # Instead of 80, 60
  ```
- Want faster results? Use CG for quick approximations
- Need high accuracy? Use ADI (it's worth the extra time)

---

## Output Files

Each notebook saves figures to:
```
02-Elliptic-Equations/figures/
```

**From notebook 04**:
- `solver_timing_comparison.png` - Bar chart of execution times
- `iteration_comparison.png` - Iteration counts
- `heatmap_comparison.png` - Temperature distributions
- `3d_surface_direct.png` - 3D visualization
- `convergence_residual_decay.png` - Convergence history
- `speedup_analysis.png` - Speedup factors
- `accuracy_vs_speed.png` - Trade-off scatter plot
- `advanced_solver_summary.csv` - Performance data table

---

## Further Reading

### Papers
- Peaceman & Rachford (1955): "The numerical solution of parabolic and elliptic differential equations" - Original ADI paper
- Douglas & Rachford (1956): "On the numerical solution of heat conduction problems"
- Young (1971): "Iterative Solution of Large Linear Systems" - SOR theory

### Books
- LeVeque (2007): "Finite Difference Methods for Ordinary and Partial Differential Equations"
- Saad (2003): "Iterative Methods for Sparse Linear Systems"
- Briggs et al. (2000): "A Multigrid Tutorial"

### Code
- Chapter 01 (`01-Linear-Systems`): Thomas algorithm implementation used in ADI/Line-SOR
- `elliptic.py`: Core solver implementations with detailed docstrings

---

## Contributing

Found a bug or have suggestions?
1. Check existing issues in the repository
2. Create a new issue with notebook name and cell number
3. Or submit a PR with fixes/improvements

---

## Citation

If you use these notebooks in your research, please cite:
```
@misc{computational-physics-notebooks,
  title={Computational Physics: Numerical Methods for Elliptic PDEs},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/Computational-Physics-Numerical-methods}
}
```

---

**Last Updated**: November 26, 2025  
**Bug Fix**: ADI and Line-SOR methods corrected (diagonal modification in operator splitting)
