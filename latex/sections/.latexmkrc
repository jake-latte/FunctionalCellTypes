# latexmk does not search parent directories for an rc file, so this one
# delegates to the shared settings in latex/.latexmkrc (output goes to ./build).
# `do` returns undef silently when the file is missing; die so a build can
# never fall back to writing beside the source.
do '../.latexmkrc' or die "cannot read ../.latexmkrc: $! $@";
