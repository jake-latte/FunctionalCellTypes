# Build the full document:        make
# Build one section standalone:   make sec SEC=01_setup
# Remove build artefacts:         make clean

MAIN := main
SEC  ?= 01_setup

.PHONY: all sec clean

all:
	latexmk -pdf $(MAIN).tex

sec:
	cd sections && latexmk -pdf $(SEC).tex

clean:
	latexmk -C $(MAIN).tex
	cd sections && latexmk -C *.tex
