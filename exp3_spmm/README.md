
# SpMM CUDA Experiment (Student ZIP)

This archive contains starter and student-skeleton files for a Sparse Matrix × Dense Matrix (SpMM) CUDA lab.

Files:
- `spmm_ref.cpp` — CPU reference implementation (complete).
- `spmm_baseline.cu` — Baseline CUDA file with TODOs for students to complete.
- `spmm_opt.cu` — Optimized kernel skeleton with TODOs (warp-based).
- `Makefile` — build targets.

How to build:
- Requires CUDA toolkit and nvcc available.
- From the archive root: `make`

Notes for instructors:
- The skeletons intentionally leave key kernel parts as TODOs for students.
- You can adjust default matrix sizes inside the .cu files for smaller/faster runs.
