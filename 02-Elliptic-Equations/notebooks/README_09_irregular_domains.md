# Notebook 09: Solving PDEs on Irregular Domains

## Overview

This comprehensive notebook demonstrates three progressively sophisticated approaches for solving elliptic PDEs on non-rectangular domains using **Cartesian grids**.

## Contents

### Part 1: L-Shaped Domains (Cells 4-14)
**Domain masking for piecewise rectangular geometries**

- **Theory**: Re-entrant corner singularities, weak solutions
- **Method**: Boolean mask + modified discrete Laplacian
- **Convergence**: **O(h²)** - optimal for aligned boundaries
- **Test cases**:
  - Laplace equation with homogeneous BC
  - Non-homogeneous Dirichlet BC
  - Poisson equation with source term

**Key functions**:
- `create_lshaped_mask()` - Generate domain mask
- `build_laplacian_masked()` - Sparse matrix assembly
- `solve_lshaped_poisson()` - Complete solver

### Part 2: Circular and Elliptical Domains (Cells 16-30)
**Staircase boundary approximation for curved boundaries**

- **Theory**: Level set representation, geometric error analysis
- **Method**: Implicit surface φ(x,y) < 0 defines interior
- **Convergence**: **O(h)** - limited by boundary approximation
- **Test cases**:
  - Laplace on circle (sanity check)
  - Poisson with radial source
  - Rotated elliptical domains
  - **Convergence study** with manufactured solution

**Key functions**:
- `create_circular_mask()` - Circle domain
- `create_elliptical_mask()` - Ellipse with rotation
- `solve_circular_poisson()` - Unified solver (reuses `build_laplacian_masked`)

**Important result**: Convergence study demonstrates O(h) bottleneck

| Grid size | L² error | Order |
|-----------|----------|-------|
| 21×21 | 1.23e-3 | - |
| 41×41 | 6.54e-4 | 1.0 |
| 81×81 | 3.28e-4 | 1.0 |
| 161×161 | 1.64e-4 | 1.0 |

### Part 3: Immersed Boundary Method (Cells 32-44)
**Ghost-cell interpolation for complex smooth geometries**

- **Theory**: Signed distance functions, ghost-cell extrapolation
- **Method**: Modified stencil near boundary, distance-based BC enforcement
- **Convergence**: **O(h) to O(h²)** achievable with careful implementation
- **Test cases**:
  - IBM vs staircase direct comparison
  - Star-shaped non-convex domains (3, 5, 6 points)
  - **Geometry gallery** - circle, ellipse, multiple stars

**Key functions**:
- `signed_distance_circle()` / `_ellipse()` / `_star()` - Implicit surfaces
- `find_boundary_points()` - Detect near-boundary points
- `build_ibm_laplacian()` - IBM-modified sparse matrix
- `solve_ibm_poisson()` - Complete IBM solver

**Advantages**:
- ✅ Smoother boundary representation
- ✅ Better convergence potential (O(h²) with refinement)
- ✅ Works for arbitrary smooth boundaries
- ✅ Same code infrastructure for all shapes

### Appendix (Cells 46-47)
**Visual summary and comparison table**

Comprehensive side-by-side comparison of all three methods with:
- Same source term across different geometries
- Convergence order comparison
- Computational cost analysis
- Practical recommendations

## Learning Objectives

By completing this notebook, you will:

✅ Understand trade-offs between **simplicity** and **accuracy**  
✅ Recognize when **O(h) vs O(h²)** convergence matters  
✅ Implement domain masking for irregular boundaries  
✅ Use signed distance functions for implicit geometry  
✅ Choose appropriate method based on problem requirements  

## Quick Start

```python
# Recommended execution: Run all cells sequentially
# Estimated time: 3-5 minutes
```

### Jump to specific content:
- **Want simple masking?** → Start at Cell 4 (Part 1)
- **Need curved boundaries?** → Jump to Cell 16 (Part 2)
- **Want production-quality?** → Go to Cell 32 (Part 3)
- **Compare all methods?** → See Cell 46 (Appendix)

## Method Comparison Table

| Method | Domain Types | Convergence | Implementation | When to Use |
|--------|-------------|-------------|----------------|-------------|
| **Masking** | Piecewise rectangular | O(h²) | Simple | L-shapes, steps, aligned boundaries |
| **Staircase** | Any geometry | **O(h)** | Simple | Prototyping, visualization, exploratory work |
| **IBM** | Smooth boundaries | O(h²) capable | Moderate | Production code, accuracy matters, moving boundaries |

## Key Insights

### 1. Boundary treatment dominates accuracy
```
Interior stencil: O(h²) ✅
      ↓
Aligned boundary:  O(h²) ✅ (Masking)
Staircase boundary: O(h) ⚠️ (Geometric error)
IBM boundary:      O(h²) ✅ (Ghost-cell interpolation)
```

### 2. O(h) vs O(h²) makes huge practical difference

To gain **3 more digits of accuracy**:
- **O(h)**: Need 1000× more grid points! 😱
- **O(h²)**: Need only 32× more points ✅

### 3. No universal best method

Choose based on:
- Required accuracy tolerance
- Boundary smoothness
- Implementation time budget
- Computational resources
- Moving vs static boundaries

## Code Reusability

All three methods share common infrastructure:

```python
# MODULAR DESIGN PATTERN:

# 1. Define geometry (mask or distance function)
mask = create_SHAPE_mask(nx, ny, ...)
# or
phi = signed_distance_SHAPE(X, Y, ...)

# 2. Build system (same function!)
A, b, interior, idx = build_laplacian_masked(nx, ny, hx, hy, mask_or_interior, bc_val)
# or for IBM:
A, b, interior, idx = build_ibm_laplacian(nx, ny, hx, hy, phi, bc_val)

# 3. Add source
b += source.flatten()[idx]

# 4. Solve
u_interior = spsolve(A, b)

# 5. Reconstruct
u = np.full((ny, nx), np.nan)
u[interior] = u_interior
```

**Extending to new geometries is trivial!** Just write a new `signed_distance_SHAPE()` function.

## Dependencies

Required:
```python
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4
```

Optional (not used in this notebook):
```python
numba  # For acceleration
```

## Generated Figures

The notebook saves all figures to `02-Elliptic-Equations/figures/`:

1. `lshaped_laplace.png` - L-shaped Laplace solution
2. `lshaped_nonhomo_bc.png` - Non-homogeneous BC
3. `lshaped_poisson.png` - Poisson with source
4. `staircase_resolution.png` - Multi-resolution comparison
5. `circle_laplace.png` - Circular domain Laplace
6. `circle_poisson.png` - Circular Poisson with radial source
7. `ellipse_solution.png` - Rotated elliptical domain
8. `staircase_convergence.png` - **Critical**: O(h) convergence demonstration
9. `geometry_comparison.png` - Multi-geometry side-by-side
10. `ibm_distance_function.png` - Distance function visualization
11. `ibm_vs_staircase.png` - Direct comparison
12. `ibm_star_domain.png` - Star-shaped complex geometry
13. `ibm_geometry_gallery.png` - Multiple shapes showcase
14. `complete_summary.png` - **Comprehensive**: All three methods compared

## Further Reading

### Classical Papers
- Peskin (1972) - Original immersed boundary method
- LeVeque & Li (1994) - Immersed interface method
- Udaykumar et al. (1997) - Ghost-cell method
- Colella et al. (2006) - Embedded boundary methods

### Modern Reviews
- Mittal & Iaccarino (2005) - *Ann. Rev. Fluid Mech.* - IBM comprehensive review
- Griffith & Peskin (2020) - *Ann. Rev. Fluid Mech.* - 50 years of IB method

### Textbooks
- LeVeque (2007) - *Finite Difference Methods for ODEs and PDEs*
- Peyret (2002) - *Spectral Methods for Incompressible Viscous Flow*

## Next Steps

### Exercises (progressively challenging)

1. **Easy**: Modify L-shape to T-shape or + shape
2. **Medium**: Implement Neumann BC for circular domain
3. **Medium**: Add 3D version (sphere, ellipsoid)
4. **Hard**: Time-dependent heat equation on moving circle
5. **Very Hard**: Higher-order IBM with quadratic extrapolation
6. **Research**: Adaptive mesh refinement guided by error estimator

### Extensions

- Variable coefficients: $-\nabla \cdot (\kappa(x,y) \nabla u) = f$
- Mixed boundary conditions (Dirichlet + Neumann)
- Discontinuous coefficients across interfaces
- Coupled systems (Stokes flow, elasticity)
- 3D problems (spheres, toruses, complex CAD geometries)

## Troubleshooting

**Problem**: "ModuleNotFoundError: No module named 'elliptic'"
- **Solution**: Ensure you run the setup cell (Cell 2) first

**Problem**: Convergence study doesn't show O(h)
- **Check**: Are you using manufactured solution with known exact answer?
- **Check**: Is boundary error actually dominating? (May need finer grids)

**Problem**: IBM produces NaN or diverges
- **Check**: Distance function has correct sign (negative inside)
- **Check**: Boundary points properly identified (not too far from boundary)

**Problem**: Solutions look wrong
- **Verify**: Boundary conditions applied correctly
- **Verify**: Source term sign (note: $-\Delta u = f$, so negative Laplacian)
- **Debug**: Start with Laplace (f=0) as sanity check

## Contact & Contribution

This notebook is part of the **Computational Physics: Numerical Methods** repository.

**Repository**: `Computational-Physics-Numerical-methods`  
**Chapter**: 02 - Elliptic Equations  
**Notebook**: 09 - Irregular Domains  

For questions, issues, or improvements:
1. Check existing documentation in `02-Elliptic-Equations/README.md`
2. Review `CHAPTER_INVENTORY.md` for full chapter overview
3. Open issue on repository (if applicable)

---

**Happy computing!** 🚀

*Last updated: 2025-11-30*
