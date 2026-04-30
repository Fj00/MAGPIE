#!/bin/bash
# Run wordstats for each phony KWG
# Usage: ./run_wordstats.sh [num_games] [threads_per_process]

GAMES=${1:-100000000}
THREADS=${2:-4}
MAGPIE_DIR="${MAGPIE_DIR:-/home/eric/MAGPIE}"

KWGS=(
  CSW24_PHONY_BASE
  CSW24_PHONY_PLUS_10
  CSW24_PHONY_PLUS_11
  CSW24_PHONY_PLUS_12
  CSW24_PHONY_PLUS_13
  CSW24_PHONY_PLUS_14
  CSW24_PHONY_PLUS_15
  CSW24_PHONY_ONLY
)

for KWG in "${KWGS[@]}"; do
  echo "=== $KWG ==="
  echo "set -lex $KWG -leaves CSW24 -s1 equity -s2 equity -r1 all -r2 all -numplays 1 -gp false -threads $THREADS -printboards false -savesettings false -wmp false
autoplay wordstats $GAMES -wb 1000000" | "$MAGPIE_DIR/bin/magpie" > "$MAGPIE_DIR/logs/${KWG}.log" 2>&1 &
  echo "Started $KWG (PID $!)"
done

echo ""
echo "All started. Monitor with: tail -f $MAGPIE_DIR/logs/*.log"
echo "Output files: $MAGPIE_DIR/data/lexica/*_word_stats.csv"
