#!/bin/bash
# Reverse the figure reorganisation recorded in MOVES.tsv (run from repo root).
while IFS=$'\t' read -r src dst; do
  [ -f "$dst" ] && mv -n "$dst" "$src"
done < FIGURES_summerjob/MOVES.tsv
