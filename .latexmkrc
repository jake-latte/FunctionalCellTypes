# latexmk configuration: build PDF with pdflatex and biber.
$pdf_mode = 1;
$bibtex_use = 2;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$clean_ext = 'synctex.gz run.xml bbl bcf fdb_latexmk fls';
