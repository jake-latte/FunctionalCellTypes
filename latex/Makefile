# Build the full document:        make            -> build/main.pdf
# Build one section standalone:   make sec SEC=01_setup   -> build/01_setup.pdf
# Remove build artefacts:         make clean

MAIN   := main
SEC    ?= 01_setup
OUTDIR := $(CURDIR)/build

.PHONY: all sec clean

all:
	latexmk -pdf -outdir=$(OUTDIR) $(MAIN).tex

sec:
	cd sections && latexmk -pdf -synctex=1 -outdir=$(OUTDIR) $(SEC).tex

clean:
	rm -rf $(OUTDIR)
