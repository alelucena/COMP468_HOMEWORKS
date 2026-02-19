#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dmlp"
LAYERS=("512,512,512" "1024,2048,1024" "2048,2048,2048")
BATCHES=(64 128 256 512)
IMPLS=(baseline activation_fused)
ACTIVATION="relu"

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_mlp_sweep.csv"
echo "impl,layers,batch,activation,time_ms,gflops,max_error" > "$LOG"

for layers in "${LAYERS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Running $impl layers=$layers batch=$batch"
      # TODO(student): parse stdout from the binary and append to the CSV.
      # 2>&1 ensures that the binary prints is caught even if it sent to stderr
      "$BIN" --layers "$layers" --batch "$batch" --activation "$ACTIVATION" --impl "$impl" --no-verify 2>&1 \
      | grep "Impl=" \
|     awk -F'[= ]' -v act="$ACTIVATION" '{print $2","$6","$4","act","$8","$10}' >> "$LOG"
    done
  done
done

echo "Results stored in $LOG"
