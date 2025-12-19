@echo off
echo Compiling 2-page extended abstract...
pdflatex -jobname=extended_abstract main.tex
bibtex extended_abstract
pdflatex -jobname=extended_abstract main.tex
pdflatex -jobname=extended_abstract main.tex
echo Done! Output: extended_abstract.pdf

