#!/bin/bash
# Script para instalar dependencias del entorno virtual

set -e

echo "=============================================="
echo "Instalando dependencias para Chapter 03"
echo "=============================================="

# Activar entorno virtual si existe
if [ -d "/workspaces/Computational-Physics-Numerical-methods/.venv" ]; then
    source /workspaces/Computational-Physics-Numerical-methods/.venv/bin/activate
    echo "✓ Entorno virtual activado"
else
    echo "⚠ No se encontró el entorno virtual en .venv"
    echo "  Creando entorno virtual..."
    python3 -m venv /workspaces/Computational-Physics-Numerical-methods/.venv
    source /workspaces/Computational-Physics-Numerical-methods/.venv/bin/activate
    echo "✓ Entorno virtual creado y activado"
fi

# Instalar ipykernel primero
echo ""
echo "Instalando ipykernel..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install ipykernel jupyter -q
echo "✓ ipykernel instalado"

# Instalar requirements del Chapter 03
echo ""
echo "Instalando requirements de 03-Eigenvalue-Problems..."
pip install -r /workspaces/Computational-Physics-Numerical-methods/03-Eigenvalue-Problems/requirements.txt -q
echo "✓ Requirements instalados"

# Instalar requirements del Chapter 01 (para inter-chapter imports)
echo ""
echo "Instalando requirements de 01-Linear-Systems..."
pip install -r /workspaces/Computational-Physics-Numerical-methods/01-Linear-Systems/requirements.txt -q
echo "✓ Requirements de Chapter 01 instalados"

echo ""
echo "=============================================="
echo "✓ Instalación completa"
echo "=============================================="
echo ""
echo "Ahora puedes:"
echo "1. Ejecutar el notebook verify_eigensolvers.ipynb"
echo "2. Ejecutar pytest: cd 03-Eigenvalue-Problems && pytest tests/ -v"
echo "3. Ejecutar script: python tests/run_verification.py"
