#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dgemm"
SIZES=(512 1024 2048 4096)
IMPLS=(cublas tiled naive)

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_sweep.csv"
echo "impl,m,n,k,time_ms,gflops" > "$LOG"

for n in "${SIZES[@]}"; do
  for impl in "${IMPLS[@]}"; do
    echo "Running $impl N=$n"
    # TODO(student): parse binary output and append to CSV (e.g., using grep/awk)
    # 2>&1 ensures that the binary prints is caught even if it sent to stderr
    "$BIN" --m "$n" --n "$n" --k "$n" --impl "$impl" --no-verify 2>&1 \
      | grep "Impl=" \
      | awk -F'[= ]' '{print $2","$4","$6","$8","$10","$12}' >> "$LOG"
  done
done

echo "Results stored in $LOG"

