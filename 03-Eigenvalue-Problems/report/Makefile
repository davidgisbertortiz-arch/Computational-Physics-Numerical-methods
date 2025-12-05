# Makefile para compilar el informe del Capítulo 03

MAIN = chapter03_eigenvalue_problems
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex  # Segunda pasada para referencias

clean:
	rm -f *.aux *.log *.toc *.out *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

cleanall: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "PDF generado: $(MAIN).pdf"

# Alias
pdf: all
