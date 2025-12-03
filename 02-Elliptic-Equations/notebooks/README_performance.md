# Performance Optimization Notebooks

## Notebook 08: Performance Optimization

This notebook demonstrates advanced performance techniques for large-scale elliptic PDE solvers:

### Part 1-4: Core Optimizations
- **Matrix-free methods** with `scipy.sparse.linalg.LinearOperator`
- **Memory comparison**: explicit vs matrix-free (1000× savings)
- **Cache-friendly line-relaxation** for row-major memory layout
- **Comprehensive benchmarks** across problem sizes

### Part 5: Parallel Iterative Solvers with Numba
- **Parallel Jacobi** with OpenMP threading
- **Red-Black SOR** for parallel convergence
- **Performance benchmarks**: 2-4× speedup on multi-core CPUs
- **Solution quality verification** against direct solver

## Installation

### Required Dependencies
```bash
pip install numpy scipy matplotlib pandas pytest
```

### Optional (for Part 5: Parallel Solvers)
```bash
pip install numba psutil
```

**Note**: Numba requires a C/C++ compiler and may need additional setup on some systems.

## Quick Start

### Running the Notebook
```bash
cd 02-Elliptic-Equations
jupyter lab notebooks/08_performance_optimization.ipynb
```

### Testing Numba Installation
```python
import numba
print(f"Numba version: {numba.__version__}")
print(f"Threading layer: {numba.config.THREADING_LAYER}")
```

### Setting Number of Threads
```bash
# Before launching Jupyter
export OMP_NUM_THREADS=4  # Use 4 threads
jupyter lab
```

Or in Python:
```python
import os
os.environ['OMP_NUM_THREADS'] = '4'
```

## Expected Performance

### Matrix-Free Memory Savings
| Grid Size | Explicit Matrix | Matrix-Free | Savings |
|-----------|----------------|-------------|---------|
| 101×101   | 1.6 MB        | ~100 bytes  | 16,000× |
| 401×401   | 25 MB         | ~100 bytes  | 250,000× |
| 1001×1001 | 400 MB        | ~100 bytes  | 4,000,000× |

### Parallel Speedup (4 cores)
| Method | Sequential | Parallel | Speedup |
|--------|-----------|----------|---------|
| Jacobi | 10.2 s    | 3.1 s    | 3.3×   |
| Red-Black SOR | 4.8 s | 1.5 s | 3.2×   |

*Results on 401×401 grid, Intel i7-8550U (4 cores)*

## Troubleshooting

### Numba Not Available
If you see "⚠️ Numba not installed", the notebook will skip Part 5 (parallel solvers) but Parts 1-4 will work fine.

Install Numba:
```bash
pip install numba
```

### Poor Parallel Performance
1. **Check thread count**:
   ```python
   import os
   print(os.environ.get('OMP_NUM_THREADS', 'not set'))
   ```

2. **Set explicitly**:
   ```bash
   export OMP_NUM_THREADS=4  # Match your CPU cores
   ```

3. **Verify Numba threading**:
   ```python
   import numba
   print(numba.config.THREADING_LAYER)  # Should be 'omp' or 'tbb'
   ```

### Memory Issues
For very large problems (> 10⁷ unknowns):
- Use matrix-free methods (Part 1)
- Reduce grid size for testing
- Monitor memory with `psutil`

## Performance Tips

1. **Start small**: Test on 51×51 grid first
2. **Profile carefully**: Use `time.time()` or `%timeit` in Jupyter
3. **Check convergence**: Verify `info['converged'] == True`
4. **Compare methods**: Different problems favor different solvers
5. **Use appropriate tolerance**: `tol=1e-6` usually sufficient

## Next Steps

After mastering this notebook, explore:
- **Advanced preconditioners** (ILU, AMG)
- **GPU acceleration** with CuPy/JAX
- **Distributed computing** with MPI
- **Adaptive mesh refinement**

## References

- **Numba documentation**: https://numba.pydata.org/
- **SciPy LinearOperator**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html
- **OpenMP**: https://www.openmp.org/

---

**Questions or issues?** Check the main repository README or open an issue on GitHub.
