# latexmk does not search parent directories for an rc file, so this one
# delegates to the shared settings in latex/.latexmkrc (output goes to ./build).
do '../.latexmkrc';
