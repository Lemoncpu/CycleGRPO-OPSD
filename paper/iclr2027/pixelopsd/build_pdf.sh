#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdfinfo main.pdf | grep '^Pages:'
