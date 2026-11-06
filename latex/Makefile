# Every build writes to a build/ directory beside its root .tex file.
# Flags and the output directory live in the .latexmkrc files, so these targets
# and a bare `latexmk` agree. Targets work from any directory
# (`make -f latex/Makefile ...`) because they are anchored on this file.
#
#   make                       main.tex              -> build/main.pdf
#   make sec SEC=01_setup      sections/01_setup.tex -> sections/build/01_setup.pdf
#   make note NOTE=<name>      notes/<name>.tex      -> notes/build/<name>.pdf
#   make notes                 every notes/*.tex     -> notes/build/
#                              (add -k to keep going past a failing note)
#   make check                 run scripts/check_build_dirs.sh
#   make clean                 removes build/, sections/build/, notes/build/

ROOT := $(dir $(lastword $(MAKEFILE_LIST)))
NOTE_NAMES := $(basename $(notdir $(wildcard $(ROOT)notes/*.tex)))
NOTE_TARGETS := $(addprefix note-,$(NOTE_NAMES))

.PHONY: all sec note notes check clean $(NOTE_TARGETS)

all:
	cd $(ROOT) && latexmk main.tex

# guard-VAR fails with a usage message when VAR is empty.
guard-%:
	@test -n "$($*)" || { echo "usage: make $(if $(filter SEC,$*),sec SEC,note NOTE)=<name>" >&2; exit 2; }

sec: guard-SEC
	cd $(ROOT)sections && latexmk $(SEC).tex

note: guard-NOTE note-$(NOTE)

notes: $(NOTE_TARGETS)

$(NOTE_TARGETS): note-%:
	cd $(ROOT)notes && latexmk $*.tex

check:
	$(ROOT)scripts/check_build_dirs.sh

clean:
	cd $(ROOT) && rm -rf build sections/build notes/build
