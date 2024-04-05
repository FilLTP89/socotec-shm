#!/bin/bash

rm -fr *.log,*.aux,*.dvi,*.lof,*.lot,*.bit,*.idx,*.glo,*.bbl,*.bcf,*.ilg,*.toc,*.ind,*.out,*.blg,*.fdb_latexmk,*.fls,*.out,*.run.xml,*.synctex.gz,*.snm,*.vrb,*.nav,*.aux,*.bcf,*.bbl,*.blg,*.log

export clatex="pdflatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error"

$clatex $1.tex 
bibtex $1.aux 
$clatex $1.tex
$clatex $1.tex

# $clatex $2.tex 
# bibtex $2.aux 
# $clatex $2.tex
# $clatex $2.tex

rm -fr *.log,*.aux,*.dvi,*.lof,*.lot,*.bit,*.idx,*.glo,*.bbl,*.bcf,*.ilg,*.toc,*.ind,*.out,*.blg,*.fdb_latexmk,*.fls,*.out,*.run.xml,*.synctex.gz,*.snm,*.vrb,*.nav,*.aux,*.bcf,*.bbl,*.blg,*.log

rm -fr ./*-eps-converted-to.pdf
