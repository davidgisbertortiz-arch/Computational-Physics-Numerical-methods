"""Script demo para el capítulo de ecuaciones elípticas.

Ejecutar desde la raíz del repo:

    python 02-Elliptic-Equations/run_elliptic.py

"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt

from elliptic import build_poisson_2d, solve_direct, solve_cg, jacobi, sor


def demo_dirichlet():
    nx, ny = 60, 40
    bc = {'left': ('dirichlet', 0.0), 'right': ('dirichlet', 0.0),
          'bottom': ('dirichlet', 0.0), 'top': ('dirichlet', 100.0)}

    A, b, meta = build_poisson_2d(nx, ny, lx=2.0, ly=1.0, bc=bc)
    print('Matrix size:', A.shape)

    # Direct solve
    x_direct = solve_direct(A, b)

    # CG
    x_cg, info = solve_cg(A, b, tol=1e-8, precondition='jacobi')
    print('CG info:', info)

    # Jacobi quick benchmark
    x_jacobi, it_j, r_j = jacobi(A, b, tol=1e-6, maxiter=5000)
    print('Jacobi iters:', it_j, 'residual:', r_j)

    nx_tot, ny_tot, hx, hy = meta
    nx_i = nx_tot - 2
    ny_i = ny_tot - 2

    def to_grid(x_vec):
        U = np.zeros((ny_tot, nx_tot))
        k = 0
        for i in range(nx_i):
            for j in range(ny_i):
                U[j+1, i+1] = x_vec[k]
                k += 1
        # set BCs
        U[0, :] = bc['bottom'][1]
        U[-1, :] = bc['top'][1]
        U[:, 0] = bc['left'][1]
        U[:, -1] = bc['right'][1]
        return U

    Ud = to_grid(x_direct)
    Uc = to_grid(x_cg)
    Uj = to_grid(x_jacobi)

    outdir = Path('02-Elliptic-Equations/figures')
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Direct')
    plt.imshow(Ud, origin='lower', cmap='coolwarm', extent=[0, 2.0, 0, 1.0])
    plt.colorbar(shrink=0.6)

    plt.subplot(1, 3, 2)
    plt.title('CG')
    plt.imshow(Uc, origin='lower', cmap='coolwarm', extent=[0, 2.0, 0, 1.0])
    plt.colorbar(shrink=0.6)

    plt.subplot(1, 3, 3)
    plt.title('Jacobi')
    plt.imshow(Uj, origin='lower', cmap='coolwarm', extent=[0, 2.0, 0, 1.0])
    plt.colorbar(shrink=0.6)

    plt.tight_layout()
    plt.savefig(outdir / 'demo_dirichlet.png', dpi=200)
    print('Saved figures to', outdir)


if __name__ == '__main__':
    demo_dirichlet()
