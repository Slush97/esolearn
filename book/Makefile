.PHONY: pdf clean

pdf:
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
	rm -f chapters/*.aux appendices/*.aux
