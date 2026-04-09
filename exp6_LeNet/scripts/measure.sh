#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dlenet"
BATCHES=(32 64 128)
ALGOS=(implicit_gemm implicit_precomp fft)
IMPLS=(baseline) #removed fused

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_lenet_sweep.csv"
echo "impl,batch,algo,time_ms,gflops,workspace_bytes" > "$LOG"

for batch in "${BATCHES[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Running $impl batch=$batch algo=$algo"
      # TODO(student): parse stdout and append to CSV (e.g., grep GFLOP/s, awk fields).
      "$BIN" --batch "$batch" --algo "$algo" --impl "$impl" --no-verify 2>&1 \
      | grep "Impl=" \
      | awk -F'[= ]' '{print $2","$4","$6","$8","$10","$12}' >> "$LOG"
    done
  done
done

echo "Results stored in $LOG"
