# Every build writes to a build/ beside the root .tex file being built.
# This is the single source of settings: latex/sections/.latexmkrc and
# latex/notes/.latexmkrc delegate here, because latexmk does not search parent
# directories for an rc file. The Makefile and a bare `latexmk <file>.tex` both
# pick it up; LaTeX Workshop gets the same result from
# `latex-workshop.latex.outDir` = %DIR%/build.
$pdf_mode = 1;
$bibtex_use = 2;
$out_dir = 'build';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
# Do not hard-wrap log lines at 79 columns, so error lines stay greppable.
$ENV{'max_print_line'} = 1000;
# The delegating rc files test this file's value with `do ... or die`.
1;
