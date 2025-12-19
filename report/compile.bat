@echo off
echo Compiling LaTeX report...
pdflatex gnn_vae_rg_report.tex
bibtex gnn_vae_rg_report
pdflatex gnn_vae_rg_report.tex
pdflatex gnn_vae_rg_report.tex
echo Done! Output: gnn_vae_rg_report.pdf
