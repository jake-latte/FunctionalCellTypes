# Every build writes to a build/ directory beside its root .tex file.
# Run from latex/. Flags and the output directory live in the .latexmkrc of
# each directory, so these targets and a bare `latexmk` agree.
#
#   make                       main.tex              -> build/main.pdf
#   make sec SEC=01_setup      sections/01_setup.tex -> sections/build/01_setup.pdf
#   make note NOTE=<name>      notes/<name>.tex      -> notes/build/<name>.pdf
#   make notes                 every notes/*.tex     -> notes/build/
#   make clean                 removes build/, sections/build/, notes/build/

SEC  ?= 01_setup
NOTE ?=
NOTE_NAMES := $(basename $(notdir $(wildcard notes/*.tex)))

.PHONY: all sec note notes clean

all:
	latexmk main.tex

sec:
	cd sections && latexmk $(SEC).tex

note:
	@test -n "$(NOTE)" || { echo "usage: make note NOTE=<name>  (a file in notes/)" >&2; exit 2; }
	cd notes && latexmk $(NOTE).tex

# -k keeps going past a failing note; make then exits non-zero if any failed.
notes:
	$(MAKE) -k $(addprefix note-,$(NOTE_NAMES))

note-%:
	$(MAKE) note NOTE=$*

clean:
	rm -rf build sections/build notes/build
