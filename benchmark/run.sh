#!/usr/bin/env bash
# Benchmark runner.
#
#   ./run.sh                       full protocol, then the table from results.json
#   ./run.sh --skip-train          part 1 only, then the table
#   ./run.sh --table-only [FILE]   regenerate the table only (default results.json)
#   PYTHON=/path/to/python ./run.sh
#
# Any other argument goes straight to bench_efficiency.py: --n, --trials, --steps,
# --reps, --batch, --data-dir, --json, --markdown, --seed, --train-only.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for cand in python3 python; do
    command -v "$cand" >/dev/null 2>&1 && { PYTHON="$cand"; break; }
  done
fi
[ -n "$PYTHON" ] || { echo "no python interpreter found; set PYTHON=/path/to/python" >&2; exit 1; }

table_only=0
want_help=0
results=""
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --table-only) table_only=1; shift
                  if [ $# -gt 0 ] && [ -f "$1" ]; then results="$1"; shift; fi ;;
    -h|--help)    want_help=1; args+=("$1"); shift ;;
    *)            args+=("$1"); shift ;;
  esac
done

# The lab environment exports an LD_LIBRARY_PATH that shadows the CUDA libraries
# torch/triton dlopen; drop it.  Harmless on machines where it is not set.
if [ "$table_only" -eq 0 ]; then
  env -u LD_LIBRARY_PATH "$PYTHON" bench_efficiency.py ${args[@]+"${args[@]}"}
fi
[ "$want_help" -eq 1 ] && exit 0
[ "$table_only" -eq 1 ] && [ -z "$results" ] && results="results.json"
[ -n "$results" ] || results="results.json"
[ -f "$results" ] || { echo "no $results -- run the benchmark first: ./run.sh" >&2; exit 1; }

env -u LD_LIBRARY_PATH "$PYTHON" make_table.py "$results"
# write the rows through a temp file so a failure cannot leave an empty table
tmp="$(mktemp)"
if env -u LD_LIBRARY_PATH "$PYTHON" make_table.py "$results" --latex > "$tmp"; then
  mv "$tmp" table_rows.tex
else
  rm -f "$tmp"; echo "make_table.py failed; table_rows.tex left untouched" >&2; exit 1
fi
echo
echo "LaTeX rows for the paper's efficiency table -> table_rows.tex"
cat table_rows.tex
