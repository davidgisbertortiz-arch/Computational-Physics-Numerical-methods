## Resumen rápido del repositorio

Este repositorio contiene una colección de ejercicios y proyectos sobre métodos numéricos en física computacional. Los capítulos disponibles son:
- **Chapter 01**: `01-Linear-Systems` — sistemas lineales, matrices tridiagonales, algoritmo de Thomas, comparaciones seriales vs paralelas
- **Chapter 02**: `02-Elliptic-Equations` — ecuaciones elípticas 2D (Poisson/Laplace), métodos iterativos, line-relaxation, ADI, multigrid

### Chapter 01: Linear Systems
**Archivos clave**
- `01-Linear-Systems/src/linear_systems.py`: funciones núcleo — construcción de matrices tridiagonales, algoritmo de Thomas (`tridiagonal_solve`), utilidades de error y residual.
- `01-Linear-Systems/src/experiments.py`: scripts de experimentación (timings, precisión) que generan figuras en `01-Linear-Systems/figures`.
- `01-Linear-Systems/src/parallel_solvers.py`: implementaciones opcionales aceleradas (Numba, CuPy) y patrones para benchmarks.
- `01-Linear-Systems/notebooks/`: 6 notebooks demostrativos (ejecutables desde la carpeta del proyecto).
- `01-Linear-Systems/requirements.txt`: dependencias sugeridas para este subproyecto.

### Chapter 02: Elliptic Equations
**Archivos clave**
- `02-Elliptic-Equations/src/elliptic.py`: módulo principal (~1200 líneas) con solvers para problemas elípticos 2D
  - `build_poisson_2d()`: construcción de operador discreto 2D usando suma de Kronecker
  - Solvers básicos: `solve_direct()`, `solve_cg()`, `jacobi()`, `sor()`
  - Solvers avanzados: `line_relaxation()`, `adi_solve()` (usan algoritmo de Thomas de Chapter 01)
  - **Multigrid**: `multigrid_solve()`, `v_cycle()`, transfer operators, Red-Black Gauss-Seidel smoother — O(n) complexity, grid-independent convergence
  - **Variable coefficients**: `build_variable_coeff_2d()` para -∇·(κ(x,y)∇u)=f con harmonic averaging
- `02-Elliptic-Equations/run_elliptic.py`: script de demostración rápida
- `02-Elliptic-Equations/tests/`: test_elliptic.py (básicos), test_advanced_solvers.py (line-SOR, ADI), test_multigrid.py (15 tests), test_variable_coefficients.py
- `02-Elliptic-Equations/notebooks/`: 8 notebooks con análisis completo, benchmarks, visualizaciones
- `02-Elliptic-Equations/requirements.txt`: numpy, scipy, matplotlib, pandas, pytest, numba, psutil
- `02-Elliptic-Equations/report/`: LaTeX report with Makefile for PDF generation

## Arquitectura y patrones relevantes

### No hay paquete Python instalado
Los módulos se organizan en `XX-ChapterName/src/` y se ejecutan con imports relativos usando `Path(__file__).parent` para robustez. **No usar `pip install -e .`** — el proyecto deliberadamente evita setuptools para mantener simplicidad pedagógica.

### Diseño modular
- **Funciones puras en módulos principales** (`linear_systems.py`, `elliptic.py`): sin dependencias de I/O, side-effects mínimos, fáciles de testear
- **Scripts auxiliares** (`experiments.py`, `run_elliptic.py`): contienen flujos de trabajo, plots, benchmarking, y escritura de archivos
- **Notebooks**: siempre incluyen setup cell al inicio con configuración de sys.path

### Patrón de imports robusto
**CRÍTICO**: Todos los archivos Python usan:
```python
from pathlib import Path
import sys

# Para scripts en XX-ChapterName/
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Para tests en XX-ChapterName/tests/
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Para notebooks (primera celda ejecutable)
from pathlib import Path
import sys
repo_root = Path.cwd()
while not (repo_root / '01-Linear-Systems').exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / 'XX-ChapterName' / 'src'))
```

### Inter-chapter imports
Chapter 02 importa utilities de Chapter 01 con try/except y flag `CHAPTER01_AVAILABLE`:
```python
try:
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / '01-Linear-Systems' / 'src'))
    from linear_systems import tridiagonal_solve, ...
    CHAPTER01_AVAILABLE = True
except ImportError:
    CHAPTER01_AVAILABLE = False
```
**Regla**: funciones que dependen de Chapter 01 deben verificar `CHAPTER01_AVAILABLE` o usar fallback.

### Convención de diagonales tridiagonales (Chapter 01)
**IMPORTANTE**: Este proyecto usa una convención específica diferente a scipy:
- `d` = diagonal principal (length n)
- `u` = **sub**diagonal inferior (length n-1) — almacenada en `A[1:, 0]`
- `o` = **super**diagonal superior (length n-1) — almacenada en `A[0, 1:]`
- `tridiagonal_solve(d, u, o, b, modify_inplace=False)` implementa el algoritmo de Thomas

**ATENCIÓN**: El orden (d, u, o) difiere de la convención scipy.linalg.solve_banded que usa (u, d, l). No intercambiar sin actualizar TODOS los call sites.

### Detección de dependencias opcionales
Usar flags booleanos para disponibilidad:
```python
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Uso condicional
if NUMBA_AVAILABLE:
    result = accelerated_function(...)
else:
    result = fallback_function(...)
```
**Dependencias opcionales comunes**: `numba` (JIT), `cupy` (GPU), `pyamg` (algebraic multigrid), `fpdf2` (PDF reports)

## Flujo de trabajo del desarrollador (comandos prácticos)

### Setup inicial
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r 01-Linear-Systems/requirements.txt
pip install -r 02-Elliptic-Equations/requirements.txt
```

### Chapter 01: Linear Systems
```bash
# Ejecutar experimentos (genera PNG en ../figures)
cd 01-Linear-Systems/src && python experiments.py

# Tests
cd 01-Linear-Systems && pytest tests/ -v

# Notebooks (cd a la carpeta del capítulo primero!)
cd 01-Linear-Systems && jupyter lab notebooks/01_introduction.ipynb
```

### Chapter 02: Elliptic Equations
```bash
# Test rápido de imports/setup
python 02-Elliptic-Equations/test_imports.py

# Demo rápida
python 02-Elliptic-Equations/run_elliptic.py

# Tests (desde raíz del repo)
cd 02-Elliptic-Equations && pytest tests/ -v

# Notebooks (cd a la carpeta del capítulo primero!)
cd 02-Elliptic-Equations && jupyter lab notebooks/04_advanced_analysis.ipynb

# Compilar report LaTeX
cd 02-Elliptic-Equations/report && make
```

### Comandos útiles de debugging
```bash
# Verificar que imports funcionan
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd() / '01-Linear-Systems' / 'src')); from linear_systems import tridiagonal_solve; print('✓ Chapter 01 OK')"

# Test rápido de solver tridiagonal
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path.cwd() / '01-Linear-Systems' / 'src')); from linear_systems import build_discrete_laplacian_1d, tridiagonal_solve; import numpy as np; d,u,o=build_discrete_laplacian_1d(10); b=np.random.randn(10); print('Solution:', tridiagonal_solve(d,u,o,b)[:3])"

# Test de available features
python 02-Elliptic-Equations/test_numba_setup.py
```

## Recomendaciones concretas para un agente de codificación (Copilot / AI)
- **No cambies firmas públicas**: funciones en `linear_systems.py` y `elliptic.py` son usadas por múltiples módulos, notebooks y tests.
- **Patrón de dependencias opcionales**: detectar disponibilidad con `try/except` y exponer flags booleanos (ej: `NUMBA_AVAILABLE`, `CUPY_AVAILABLE`, `CHAPTER01_AVAILABLE`).
- **Inter-chapter imports**: cuando un capítulo use utilities de otro, siempre incluir try/except con fallback o mensaje claro.
- **Convenciones de código**:
  - Usar docstrings NumPy-style (ejemplos: `tridiagonal_solve`, `build_discrete_laplacian_1d`)
  - Para reproducibilidad en experiments, proporcionar parámetros como argumentos en `if __name__ == '__main__'`
  - Funciones puras en módulos principales, I/O y visualización en scripts auxiliares
- **Sign conventions**: al reutilizar operadores discretos (Laplacian, etc.), verificar que las escalas (ej: 1/h²) y signos sean consistentes entre módulos.
- **scipy.sparse.linalg.cg**: usa `atol=` o `rtol=`, NO `tol=` (error común).

## Consideraciones de integración y dependencias
- Dependencias principales: `numpy`, `matplotlib`, `scipy`, `jupyter`, `pytest` (ver `01-Linear-Systems/requirements.txt`).
- Opcionales (no instaladas por defecto): `numba` (JIT/parallel), `cupy` (GPU). Indicar en notas de PR si se añaden cambios que requieren CUDA o Numba.

## Ejemplos concretos (copiar/pegar)

### Chapter 01
- Ejecutar un experimento rápido con tamaños pequeños:
  ```bash
  cd 01-Linear-Systems/src && python -c "from experiments import experiment_timing_comparison; print(experiment_timing_comparison([100,200,500]))"
  ```
- Usar el solver tridiagonal desde Python interactivo:
  ```bash
  python -c "from linear_systems import build_discrete_laplacian_1d, tridiagonal_solve; d,u,o=build_discrete_laplacian_1d(10); import numpy as np; b=np.random.randn(10); print(tridiagonal_solve(d,u,o,b))"
  ```

### Chapter 02
- Resolver problema de Poisson 2D pequeño:
  ```bash
  python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd() / '02-Elliptic-Equations' / 'src')); from elliptic import build_poisson_2d, solve_direct; import numpy as np; A,b = build_poisson_2d(10,10,1.0,1.0); x = solve_direct(A,b); print(f'Solution shape: {x.shape}, max value: {x.max():.4f}')"
  ```
- Comparar solvers con residual:
  ```bash
  cd 02-Elliptic-Equations/src && python -c "from elliptic import *; import numpy as np; A,b = build_poisson_2d(20,20,1.0,1.0); x_dir = solve_direct(A,b); x_cg,_ = solve_cg(A,b,tol=1e-6); print(f'CG vs Direct error: {np.linalg.norm(x_cg-x_dir):.2e}')"
  ```

## Qué evitar / señales de alerta
- ❌ **No mover funciones entre módulos** sin actualizar imports en TODOS los archivos dependientes (notebooks, tests, scripts).
- ❌ **No asumir dependencias opcionales disponibles**: siempre verificar flags como `NUMBA_AVAILABLE`, `CUPY_AVAILABLE`.
- ❌ **No usar `tol=` con scipy.sparse.linalg.cg**: el parámetro correcto es `atol=` o `rtol=`.
- ❌ **No modificar sign conventions** de operadores discretos sin validar con tests: errores de signo causan NaN o divergencia.
- ❌ **No olvidar Path-based imports**: usar `Path(__file__).parent` en lugar de strings hardcodeados.
- ❌ **No ejecutar notebooks sin configurar sys.path**: todos los notebooks ahora incluyen setup cell al inicio.

## Señales de que algo anda mal
- Tests que pasan pero dan valores NaN → revisar sign conventions en operadores
- `ModuleNotFoundError` → falta configurar sys.path con Path(__file__)
- `TypeError` con scipy.sparse.linalg.cg → usar `atol=` en lugar de `tol=`
- Tests que fallan después de refactoring → buscar imports rotos en notebooks/scripts auxiliares

---
Si quieres, puedo expandir esto con instrucciones de CI, plantillas de PR, o ejemplos de tests/benchmarks concretos.
