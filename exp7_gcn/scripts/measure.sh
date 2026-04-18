#!/usr/bin/env bash
set -uo pipefail 

BIN="../bin/dgcn"
GRAPHS=("data/cora")
HIDDENS=(64 128 256)
IMPLS=(baseline fused)
LAYERS=2

LOG="../data/sweep_debug.csv"
echo "graph,hidden,impl,time_ms,edges_per_s" > "$LOG"

for graph in "${GRAPHS[@]}"; do
  for hidden in "${HIDDENS[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "--- Testing: $impl | Graph: $graph | Hidden: $hidden ---"
      
      # Run the command 
      OUT=$("$BIN" --graph "../$graph" --hidden "$hidden" --layers "$LAYERS" --impl "$impl" --no-verify 2>&1)
      
      # Check if the run was successful
      if echo "$OUT" | grep -q "Impl="; then
          echo "Success."
          echo "$OUT" | grep "Impl=" | awk -F'[= ]' '{print $4","$6","$2","$10","$12}' >> "$LOG"
      else
          echo "ERROR ENCOUNTERED:"
          echo "------------------------------------------"
          echo "$OUT" # will show the actual CUDA or C++ error
          echo "------------------------------------------"
      fi
    done
  done
done