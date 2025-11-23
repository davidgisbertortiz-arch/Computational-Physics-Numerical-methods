"""
Script to execute Jupyter notebooks and save them with outputs.
"""

import subprocess
import sys
from pathlib import Path

def execute_notebook(notebook_path):
    """Execute a Jupyter notebook using nbconvert."""
    print(f"Executing {notebook_path}...")
    try:
        result = subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            '--ExecutePreprocessor.timeout=300',
            str(notebook_path)
        ], check=True, capture_output=True, text=True)
        print(f"✓ Successfully executed {notebook_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error executing {notebook_path.name}")
        print(e.stderr)
        return False

if __name__ == '__main__':
    # Ensure we're in the right directory
    project_dir = Path(__file__).parent
    notebooks_dir = project_dir / 'notebooks'
    
    # Create figures directory
    figures_dir = project_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Execute notebooks
    notebooks = [
        notebooks_dir / '01_introduction.ipynb',
        notebooks_dir / '02_thomas_algorithm.ipynb'
    ]
    
    print("=" * 70)
    print("Executing Jupyter Notebooks".center(70))
    print("=" * 70)
    
    results = []
    for notebook in notebooks:
        if notebook.exists():
            success = execute_notebook(notebook)
            results.append((notebook.name, success))
        else:
            print(f"✗ Notebook not found: {notebook}")
            results.append((notebook.name, False))
    
    print("\n" + "=" * 70)
    print("Summary".center(70))
    print("=" * 70)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{name:40} {status}")
    
    all_success = all(success for _, success in results)
    sys.exit(0 if all_success else 1)
