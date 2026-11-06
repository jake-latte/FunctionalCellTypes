#!/usr/bin/env bash
# Check that every LaTeX build writes into a build/ directory beside its root
# .tex file, and that nothing is left next to a source.
#
#   latex/main.tex            -> latex/build/
#   latex/sections/<sec>.tex  -> latex/sections/build/
#   latex/notes/<note>.tex    -> latex/notes/build/
#
# Run from anywhere:  bash latex/scripts/check_build_dirs.sh   (or: make check)
# Exits non-zero on the first violated expectation, printing the tail of the
# relevant build log. It also fails if any note's log contains a LaTeX error
# (nonstopmode still writes a PDF after an error, so a PDF is not proof).
#
# The three build/ directories are deleted at the start and on success; on
# failure they are left in place for debugging. A scratch subfile
# (sections/zz_cite_check.tex, used to exercise biber) is always removed on exit.
#
# `set -u` without -e/pipefail on purpose: every step is guarded explicitly, and
# grep returning 1 on "no match" must not abort the script.
set -u

LATEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LATEX_DIR" || exit 1

BUILD_DIRS=(build sections/build notes/build)
# Extensions a LaTeX run leaves behind (mirrors the repo .gitignore).
ARTEFACTS='\.(aux|bbl|bcf|blg|fdb_latexmk|fls|log|out|run\.xml|synctex\.gz|toc|lof|lot|nav|snm|vrb|xdv|pdf)$|-SAVE-ERROR$'
SCRATCH_NAME=zz_cite_check
SCRATCH_TEX="sections/$SCRATCH_NAME.tex"

LOG_DIR="$(mktemp -d)" || exit 1
cleanup() {
  rm -rf "$LOG_DIR" "$LATEX_DIR/$SCRATCH_TEX" "$LATEX_DIR"/sections/build/"$SCRATCH_NAME".*
}
trap cleanup EXIT

# fail MESSAGE [LOGFILE]: report, show the end of the log if given, exit 1.
fail() {
  echo "FAIL: $1" >&2
  if [ -n "${2:-}" ] && [ -f "$2" ]; then
    echo "--- last 30 lines of $2" >&2
    tail -n 30 "$2" >&2
  fi
  exit 1
}

# run NAME CMD...: run CMD with its output in $LOG_DIR/NAME.log; fail if it fails.
run() {
  local name="$1"; shift
  "$@" >"$LOG_DIR/$name.log" 2>&1 || fail "$name ($*)" "$LOG_DIR/$name.log"
}

remove_build_dirs() { rm -rf "${BUILD_DIRS[@]}"; }

# Artefact files under latex/ outside build/ (figures/ may hold tracked-by-hand
# PDFs and scripts/ may hold __pycache__; neither is a LaTeX build product).
stray_artefacts() {
  find . \( -name build -o -name figures -o -name __pycache__ \) -prune -o -type f -print \
    | grep -E "$ARTEFACTS" || true
}

assert_no_strays() {
  local strays
  strays="$(stray_artefacts)"
  [ -z "$strays" ] || fail "$1: artefacts outside build/:"$'\n'"$strays"
}

# Ignored build/ entries must be exactly the build/ dirs that exist.
assert_only_build_dirs_ignored() {
  local ignored expected d
  ignored="$(git status --short --ignored . | grep '^!!' | sed 's/^!! //' \
    | grep -E '(^|/)build/$' | sort)"
  expected="$(for d in "${BUILD_DIRS[@]}"; do [ -d "$d" ] && echo "$d/"; done | sort)"
  [ "$ignored" = "$expected" ] || fail "$1: ignored build entries differ from the build dirs:"$'\n'"got:      $ignored"$'\n'"expected: $expected"
}

assert_file() { [ -s "$1" ] || fail "missing or empty: $1"; }

# A LaTeX hard error shows as "! ..." or, with -file-line-error, "file.tex:N: ...".
assert_notes_error_free() {
  local log
  for log in notes/build/*.log; do
    if grep -qE '^!|^[^:]+\.tex:[0-9]+:' "$log"; then
      fail "note $(basename "${log%.log}") has LaTeX errors" "$log"
    fi
  done
}

# Build a subfile that cites every bib entry and check biber reads references.bib.
assert_biber_resolves() {
  cat >"$SCRATCH_TEX" <<'EOF'
\documentclass[../main.tex]{subfiles}
\begin{document}
\nocite{*}
\printbibliography
\end{document}
EOF
  run cite_build make sec SEC="$SCRATCH_NAME"
  grep -q "Found BibTeX data source" "sections/build/$SCRATCH_NAME.blg" \
    || fail "biber did not find references.bib" "sections/build/$SCRATCH_NAME.blg"
  ! grep -q "Please (re)run Biber" "sections/build/$SCRATCH_NAME.log" \
    || fail "cite check asks to rerun Biber" "sections/build/$SCRATCH_NAME.log"
  rm -f "$SCRATCH_TEX" sections/build/"$SCRATCH_NAME".*
}

echo "== make builds"
remove_build_dirs
run make make
assert_file build/main.pdf
run make_sec make sec SEC=01_setup
assert_file sections/build/01_setup.pdf
run make_note make note NOTE=continuum_limit
assert_file notes/build/continuum_limit.pdf

echo "== make notes"
run make_notes make -k notes
for tex in notes/*.tex; do
  assert_file "notes/build/$(basename "${tex%.tex}").pdf"
done
assert_notes_error_free

echo "== biber"
assert_biber_resolves
! grep -q "Please (re)run Biber" build/main.log || fail "build/main.log asks to rerun Biber" build/main.log

assert_no_strays "after make builds"
assert_only_build_dirs_ignored "after make builds"

echo "== make clean removes exactly the build dirs"
run make_clean make clean
for d in "${BUILD_DIRS[@]}"; do
  [ ! -e "$d" ] || fail "make clean left $d"
done
assert_no_strays "after make clean"

echo "== bare latexmk from each directory"
remove_build_dirs
run latexmk_main bash -c 'cd "$1" && latexmk main.tex' _ "$LATEX_DIR"
assert_file build/main.pdf
run latexmk_sec bash -c 'cd "$1/sections" && latexmk 01_setup.tex' _ "$LATEX_DIR"
assert_file sections/build/01_setup.pdf
run latexmk_note bash -c 'cd "$1/notes" && latexmk continuum_limit.tex' _ "$LATEX_DIR"
assert_file notes/build/continuum_limit.pdf
assert_no_strays "after bare latexmk"
assert_only_build_dirs_ignored "after bare latexmk"

remove_build_dirs
echo "PASS: all builds land in build/ beside their source"
