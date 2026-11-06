# Every build in this directory writes to ./build (beside the source).
# latexmk reads only the rc file in its current directory, so latex/,
# latex/sections/ and latex/notes/ each carry this same small file.
# The Makefile and a bare `latexmk <file>.tex` both pick it up; LaTeX Workshop
# gets the same result from `latex-workshop.latex.outDir` = %DIR%/build.
$pdf_mode = 1;
$bibtex_use = 2;
$out_dir = 'build';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
