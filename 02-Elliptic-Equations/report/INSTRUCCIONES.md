# Instrucciones para Compilar el Reporte LaTeX

## Método Rápido (Recomendado)

Ejecuta estos comandos en tu terminal:

```bash
cd /workspaces/Computational-Physics-Numerical-methods/02-Elliptic-Equations/report
chmod +x compile_report.sh
bash compile_report.sh
```

El script instalará LaTeX automáticamente y generará el PDF.

## Método Manual

Si prefieres hacerlo paso a paso:

```bash
# Paso 1: Instalar LaTeX (solo la primera vez)
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended

# Paso 2: Ir al directorio del reporte
cd /workspaces/Computational-Physics-Numerical-methods/02-Elliptic-Equations/report

# Paso 3: Compilar (dos veces para referencias cruzadas)
pdflatex chapter02_elliptic_equations.tex
pdflatex chapter02_elliptic_equations.tex

# Paso 4 (opcional): Limpiar archivos auxiliares
rm -f *.aux *.log *.out *.toc
```

## Resultado

Obtendrás el archivo **`chapter02_elliptic_equations.pdf`** con el reporte completo.

## ¿Qué Contiene el Reporte?

- Formulación matemática de ecuaciones elípticas 2D
- Implementación de 6 métodos numéricos diferentes
- Análisis de rendimiento y precisión
- Formulación tensorial (enfoque educativo)
- Gráficas y benchmarks de notebooks
- Código de ejemplo y algoritmos

## Solución de Problemas

Si hay errores:
1. Verifica que LaTeX esté instalado: `which pdflatex`
2. Lee los mensajes de error en la terminal
3. Asegúrate de estar en el directorio correcto: `pwd`
4. Revisa el archivo `chapter02_elliptic_equations.log` para detalles
