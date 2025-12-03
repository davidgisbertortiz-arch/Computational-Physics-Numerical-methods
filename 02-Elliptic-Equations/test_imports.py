#!/usr/bin/env python3
"""Quick test to verify imports work correctly."""

import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Try importing
try:
    from elliptic import build_poisson_2d, solve_direct, solve_cg
    print("✓ Successfully imported elliptic module")
    
    # Test a small problem
    import numpy as np
    A, b, meta = build_poisson_2d(5, 5, 1.0, 1.0, bc='dirichlet')
    print(f"✓ Built Poisson matrix: shape {A.shape}")
    
    x_direct = solve_direct(A, b)
    print(f"✓ Direct solver worked: solution shape {x_direct.shape}")
    
    x_cg, info = solve_cg(A, b, tol=1e-6, precondition=None)
    print(f"✓ CG solver worked: converged with info={info}")
    
    error = np.linalg.norm(x_direct - x_cg)
    print(f"✓ Direct vs CG error: {error:.2e}")
    
    print("\n✅ All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
