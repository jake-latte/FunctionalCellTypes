# latexmk configuration: build PDF with pdflatex and biber, output into build/.
# (Read only when latexmk runs from this directory; the Makefile passes the
# same -outdir for standalone section builds run from sections/.)
$pdf_mode = 1;
$bibtex_use = 2;
$out_dir = 'build';
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$clean_ext = 'synctex.gz run.xml bbl bcf fdb_latexmk fls';
