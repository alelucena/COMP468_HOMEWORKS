#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dgcn"
GRAPHS=("data/cora" "data/citeseer")
HIDDENS=(64 128 256)
IMPLS=(baseline fused)
LAYERS=2

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_gcn_sweep.csv"
echo "graph,hidden,impl,time_ms,edges_per_s" > "$LOG"

for graph in "${GRAPHS[@]}"; do
  for hidden in "${HIDDENS[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Running $impl graph=$graph hidden=$hidden"
      # TODO(student): parse stdout and append to CSV using awk or python -c helper.
      "$BIN" --graph "$graph" --hidden "$hidden" --layers "$LAYERS" --impl "$impl" --no-verify 2>&1 \
      | grep "Impl=" \
      | awk -F'[= ]' '{print $2","$4","$6","$8","$10","$12}' >> "$LOG"
    done
  done
done

echo "Results stored in $LOG"
