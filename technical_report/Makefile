# Build the technical report: `make` -> main.pdf ; `make view` -> open it ; `make arxiv` -> arxiv.tar.gz
main.pdf: main.tex references.bib
	latexmk -pdf -interaction=nonstopmode main.tex

view: main.pdf
	@if command -v open >/dev/null; then open main.pdf; else xdg-open main.pdf; fi

arxiv: main.pdf
	tar czf arxiv.tar.gz main.tex main.bbl references.bib

clean:
	latexmk -C
	rm -f arxiv.tar.gz

.PHONY: arxiv view clean
