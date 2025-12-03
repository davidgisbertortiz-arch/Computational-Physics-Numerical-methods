#!/bin/bash
# Script para instalar LaTeX y compilar el reporte
# Uso: bash compile_report.sh

set -e  # Salir en caso de error

echo "=== Instalando LaTeX ==="
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended

echo ""
echo "=== Compilando reporte (primera pasada) ==="
pdflatex chapter02_elliptic_equations.tex

echo ""
echo "=== Compilando reporte (segunda pasada para referencias) ==="
pdflatex chapter02_elliptic_equations.tex

echo ""
echo "=== Limpiando archivos auxiliares ==="
rm -f *.aux *.log *.out *.toc

echo ""
echo "✅ Reporte compilado exitosamente: chapter02_elliptic_equations.pdf"
ls -lh chapter02_elliptic_equations.pdf
