#!/usr/bin/env bash
# Check that every LaTeX build writes into a build/ directory beside its root
# .tex file, and that nothing is left next to a source.
#
#   latex/main.tex            -> latex/build/
#   latex/sections/<sec>.tex  -> latex/sections/build/
#   latex/notes/<note>.tex    -> latex/notes/build/
#
# Run from anywhere:  bash latex/scripts/check_build_dirs.sh
# Exits non-zero on the first violated expectation. Starts and ends with the
# three build/ directories removed, so it never depends on earlier builds.
# A note that fails to compile is reported as a warning, not a failure: note
# content is outside this check's scope.
set -u

LATEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LATEX_DIR" || exit 1

BUILD_DIRS=(build sections/build notes/build)
# Extensions a LaTeX run leaves behind (mirrors the repo .gitignore).
ARTEFACTS='\.(aux|bbl|bcf|blg|fdb_latexmk|fls|log|out|run\.xml|synctex\.gz|toc|lof|lot|nav|snm|vrb|xdv|pdf)$|-SAVE-ERROR$'

fail() { echo "FAIL: $*" >&2; exit 1; }

remove_build_dirs() { rm -rf "${BUILD_DIRS[@]}"; }

# Artefact files anywhere under latex/ that are not inside a build/ directory.
stray_artefacts() {
  find . -path '*/build' -prune -o -type f -print | grep -E "$ARTEFACTS" || true
}

assert_no_strays() {
  local strays
  strays="$(stray_artefacts)"
  [ -z "$strays" ] || fail "$1: artefacts outside build/:"$'\n'"$strays"
}

# `git status --ignored` must list exactly the build/ dirs that exist, nothing else.
assert_only_build_dirs_ignored() {
  local ignored expected
  ignored="$(git status --short --ignored . | grep '^!!' | sed 's/^!! //' | sort)"
  expected="$(for d in "${BUILD_DIRS[@]}"; do [ -d "$d" ] && echo "$d/"; done | sort)"
  [ "$ignored" = "$expected" ] || fail "$1: ignored entries differ from the build dirs:"$'\n'"got:      $ignored"$'\n'"expected: $expected"
}

assert_file() { [ -s "$1" ] || fail "missing or empty: $1"; }

echo "== make clean, then make builds"
remove_build_dirs
make >/dev/null 2>&1 || fail "make"
assert_file build/main.pdf
make sec SEC=01_setup >/dev/null 2>&1 || fail "make sec SEC=01_setup"
assert_file sections/build/01_setup.pdf
make note NOTE=continuum_limit >/dev/null 2>&1 || fail "make note NOTE=continuum_limit"
assert_file notes/build/continuum_limit.pdf

echo "== make notes"
if ! make notes >/dev/null 2>&1; then
  echo "WARN: make notes reported failing notes (see 'make notes' output)" >&2
fi
for tex in notes/*.tex; do
  pdf="notes/build/$(basename "${tex%.tex}").pdf"
  [ -s "$pdf" ] || echo "WARN: note did not compile: $tex" >&2
done

assert_no_strays "after make builds"
assert_only_build_dirs_ignored "after make builds"
if grep -q "Please (re)run Biber" build/main.log; then
  fail "build/main.log still asks to rerun Biber"
fi

echo "== make clean removes exactly the build dirs"
make clean >/dev/null 2>&1 || fail "make clean"
for d in "${BUILD_DIRS[@]}"; do
  [ ! -e "$d" ] || fail "make clean left $d"
done
assert_no_strays "after make clean"

echo "== bare latexmk from each directory"
remove_build_dirs
(cd "$LATEX_DIR" && latexmk main.tex >/dev/null 2>&1) || fail "bare latexmk main.tex"
assert_file build/main.pdf
(cd "$LATEX_DIR/sections" && latexmk 01_setup.tex >/dev/null 2>&1) || fail "bare latexmk 01_setup.tex"
assert_file sections/build/01_setup.pdf
(cd "$LATEX_DIR/notes" && latexmk continuum_limit.tex >/dev/null 2>&1) || fail "bare latexmk continuum_limit.tex"
assert_file notes/build/continuum_limit.pdf
assert_no_strays "after bare latexmk"
assert_only_build_dirs_ignored "after bare latexmk"

remove_build_dirs
echo "PASS: all builds land in build/ beside their source"
