"""
Numerical Experiments for Linear Systems

This module contains functions to run timing and accuracy experiments
comparing different solution methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict
from pathlib import Path

from linear_systems import (
    build_tridiagonal,
    build_discrete_laplacian_1d,
    tridiagonal_solve,
    compute_residual,
    compute_relative_error
)


def experiment_conditioning(sizes: List[int], num_trials: int = 5) -> Dict:
    """
    Experiment on conditioning of random matrices.
    
    Parameters
    ----------
    sizes : list of int
        Matrix sizes to test
    num_trials : int
        Number of trials per size
    
    Returns
    -------
    dict
        Results containing condition numbers and errors
    """
    results = {
        'sizes': sizes,
        'condition_numbers': [],
        'residuals': [],
        'perturbed_errors': []
    }
    
    for n in sizes:
        conds = []
        resids = []
        pert_errs = []
        
        for _ in range(num_trials):
            A = np.random.randn(n, n)
            b = np.random.randn(n)
            
            # Condition number
            cond = np.linalg.cond(A)
            conds.append(cond)
            
            # Solve and compute residual
            x = np.linalg.solve(A, b)
            resid = compute_residual(A, x, b)
            resids.append(resid)
            
            # Perturb b and measure solution change
            perturbation = 1e-8 * np.random.randn(n)
            b_perturbed = b + perturbation
            x_perturbed = np.linalg.solve(A, b_perturbed)
            
            rel_error = np.linalg.norm(x_perturbed - x) / np.linalg.norm(x)
            pert_errs.append(rel_error)
        
        results['condition_numbers'].append(np.mean(conds))
        results['residuals'].append(np.mean(resids))
        results['perturbed_errors'].append(np.mean(pert_errs))
    
    return results


def experiment_timing_comparison(sizes: List[int]) -> Dict:
    """
    Compare timing of Thomas algorithm vs numpy.linalg.solve.
    
    Parameters
    ----------
    sizes : list of int
        System sizes to test
    
    Returns
    -------
    dict
        Timing results for both methods
    """
    results = {
        'sizes': sizes,
        'thomas_times': [],
        'numpy_times': []
    }
    
    for n in sizes:
        print(f"Testing n = {n}...")
        
        # Build tridiagonal system (Laplacian-like)
        d, u, o = build_discrete_laplacian_1d(n)
        b = np.random.randn(n)
        
        # Time Thomas algorithm
        start = time.perf_counter()
        x_thomas = tridiagonal_solve(d, u, o, b)
        thomas_time = time.perf_counter() - start
        
        # Time numpy.linalg.solve
        A = build_tridiagonal(d, u, o)
        start = time.perf_counter()
        x_numpy = np.linalg.solve(A, b)
        numpy_time = time.perf_counter() - start
        
        results['thomas_times'].append(thomas_time)
        results['numpy_times'].append(numpy_time)
        
        # Verify solutions match
        error = compute_relative_error(x_thomas, x_numpy)
        if error > 1e-10:
            print(f"  Warning: Solutions differ by {error:.2e}")
    
    return results


def experiment_accuracy_vs_size(sizes: List[int]) -> Dict:
    """
    Study accuracy of tridiagonal solver vs system size.
    
    Creates systems with known exact solutions.
    
    Parameters
    ----------
    sizes : list of int
        System sizes to test
    
    Returns
    -------
    dict
        Accuracy metrics
    """
    results = {
        'sizes': sizes,
        'residuals': [],
        'relative_errors': [],
        'condition_numbers': []
    }
    
    for n in sizes:
        # Build system with known solution
        d, u, o = build_discrete_laplacian_1d(n)
        A = build_tridiagonal(d, u, o)
        
        # Create exact solution and compute b
        x_exact = np.sin(np.pi * np.linspace(1/(n+1), n/(n+1), n))
        b = A @ x_exact
        
        # Solve
        x_computed = tridiagonal_solve(d, u, o, b)
        
        # Compute metrics
        resid = compute_residual(A, x_computed, b)
        rel_err = compute_relative_error(x_computed, x_exact)
        cond = np.linalg.cond(A)
        
        results['residuals'].append(resid)
        results['relative_errors'].append(rel_err)
        results['condition_numbers'].append(cond)
    
    return results


def plot_timing_comparison(results: Dict, output_dir: str = '../figures'):
    """
    Plot timing comparison between methods.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale
    ax1.plot(results['sizes'], results['thomas_times'], 'o-', label='Thomas Algorithm')
    ax1.plot(results['sizes'], results['numpy_times'], 's-', label='numpy.linalg.solve')
    ax1.set_xlabel('System Size (n)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Timing Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale
    ax2.loglog(results['sizes'], results['thomas_times'], 'o-', label='Thomas Algorithm')
    ax2.loglog(results['sizes'], results['numpy_times'], 's-', label='numpy.linalg.solve')
    ax2.set_xlabel('System Size (n)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Timing Comparison (Log-Log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/timing_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved timing comparison to {output_dir}/timing_comparison.png")


def plot_accuracy_analysis(results: Dict, output_dir: str = '../figures'):
    """
    Plot accuracy metrics.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals
    axes[0, 0].semilogy(results['sizes'], results['residuals'], 'o-')
    axes[0, 0].set_xlabel('System Size (n)')
    axes[0, 0].set_ylabel('Residual ||Ax - b||₂')
    axes[0, 0].set_title('Residual vs Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Relative errors
    axes[0, 1].semilogy(results['sizes'], results['relative_errors'], 'o-', color='C1')
    axes[0, 1].set_xlabel('System Size (n)')
    axes[0, 1].set_ylabel('Relative Error')
    axes[0, 1].set_title('Relative Error vs Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Condition numbers
    axes[1, 0].semilogy(results['sizes'], results['condition_numbers'], 'o-', color='C2')
    axes[1, 0].set_xlabel('System Size (n)')
    axes[1, 0].set_ylabel('Condition Number κ(A)')
    axes[1, 0].set_title('Condition Number vs Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error vs condition number
    axes[1, 1].loglog(results['condition_numbers'], results['relative_errors'], 'o', color='C3')
    axes[1, 1].set_xlabel('Condition Number κ(A)')
    axes[1, 1].set_ylabel('Relative Error')
    axes[1, 1].set_title('Error vs Conditioning')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved accuracy analysis to {output_dir}/accuracy_analysis.png")


if __name__ == '__main__':
    print("=" * 60)
    print("Linear Systems Numerical Experiments")
    print("=" * 60)
    
    # Timing comparison
    print("\n1. Timing Comparison Experiment")
    print("-" * 60)
    sizes_timing = [100, 200, 500, 1000, 2000, 5000, 10000]
    timing_results = experiment_timing_comparison(sizes_timing)
    plot_timing_comparison(timing_results)
    
    # Accuracy analysis
    print("\n2. Accuracy Analysis Experiment")
    print("-" * 60)
    sizes_accuracy = [10, 20, 50, 100, 200, 500, 1000, 2000]
    accuracy_results = experiment_accuracy_vs_size(sizes_accuracy)
    plot_accuracy_analysis(accuracy_results)
    
    print("\n" + "=" * 60)
    print("Experiments completed!")
    print("=" * 60)
