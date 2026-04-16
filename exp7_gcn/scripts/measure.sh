#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dgcn"
GRAPHS=("data/cora")
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


#!/usr/bin/env bash
# REMOVE 'set -e' so the script doesn't die on a crash
# set -uo pipefail 

# BIN="../bin/dgcn"
# GRAPHS=("data/cora")
# HIDDENS=(64 128 256)
# IMPLS=(baseline fused)
# LAYERS=2

# LOG="../data/sweep_debug.csv"
# echo "graph,hidden,impl,time_ms,edges_per_s" > "$LOG"

# for graph in "${GRAPHS[@]}"; do
#   for hidden in "${HIDDENS[@]}"; do
#     for impl in "${IMPLS[@]}"; do
#       echo "--- Testing: $impl | Graph: $graph | Hidden: $hidden ---"
      
#       # Run the command and capture EVERYTHING (stdout and stderr)
#       # We don't pipe to grep yet so we can see errors
#       RAW_OUT=$("$BIN" --graph "../$graph" --hidden "$hidden" --layers "$LAYERS" --impl "$impl" --no-verify 2>&1)
      
#       # Check if the run was successful
#       if echo "$RAW_OUT" | grep -q "Impl="; then
#           echo "Success."
#           echo "$RAW_OUT" | grep "Impl=" | awk -F'[= ]' '{print $4","$6","$2","$10","$12}' >> "$LOG"
#       else
#           echo "FATAL ERROR ENCOUNTERED:"
#           echo "------------------------------------------"
#           echo "$RAW_OUT" # This will show the actual CUDA or C++ error
#           echo "------------------------------------------"
#           # Optional: exit 1 if you want to stop after seeing the first error
#       fi
#     done
#   done
# done