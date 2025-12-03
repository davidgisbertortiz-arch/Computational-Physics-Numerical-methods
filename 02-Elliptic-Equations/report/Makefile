# Makefile for Chapter 02 Report
#
# Usage:
#   make          - Compile main LaTeX report
#   make theory   - Compile mathematical theory document
#   make clean    - Remove auxiliary files
#   make view     - Open PDF (Linux/Mac)
#   make all      - Clean and compile both documents

MAIN = chapter02_elliptic_equations
THEORY = mathematical_theory
TEX = $(MAIN).tex
PDF = $(MAIN).pdf
THEORY_TEX = $(THEORY).tex
THEORY_PDF = $(THEORY).pdf

# LaTeX compiler
LATEX = pdflatex

# Auxiliary files to clean
AUX_EXT = aux log out toc bbl blg lof lot

.PHONY: all clean view help theory

# Default target
all: clean $(PDF) $(THEORY_PDF)

# Compile main report (run twice for references)
$(PDF): $(TEX)
	@echo "Compiling main report..."
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)
	@echo "Done! Output: $(PDF)"

# Compile mathematical theory document
theory: $(THEORY_PDF)

$(THEORY_PDF): $(THEORY_TEX)
	@echo "Compiling mathematical theory document..."
	$(LATEX) $(THEORY_TEX)
	$(LATEX) $(THEORY_TEX)
	@echo "Done! Output: $(THEORY_PDF)"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@for ext in $(AUX_EXT); do \
		rm -f $(MAIN).$$ext; \
		rm -f $(THEORY).$$ext; \
	done
	@echo "Clean complete"

# View PDF
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "Cannot find PDF viewer (xdg-open or open)"; \
	fi

# View theory PDF
view-theory: $(THEORY_PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(THEORY_PDF); \
	elif command -v open > /dev/null; then \
		open $(THEORY_PDF); \
	else \
		echo "Cannot find PDF viewer (xdg-open or open)"; \
	fi

# Help message
help:
	@echo "Chapter 02 Report - Makefile Targets:"
	@echo "  make          - Compile both documents"
	@echo "  make theory   - Compile mathematical theory document"
	@echo "  make all      - Clean and compile all"
	@echo "  make clean    - Remove auxiliary files (.aux, .log, etc.)"
	@echo "  make view     - Open compiled PDF"
	@echo "  make help     - Show this message"
