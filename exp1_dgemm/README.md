## Dense GEMM GPU Experiment

This mini-lab guides senior undergraduates through implementing a dense General Matrix Multiplication (GEMM) on NVIDIA GPUs using CUDA. Students start from a runnable baseline that allocates matrices, seeds data, and measures performance, then complete key CUDA kernels and host orchestration code while benchmarking against cuBLAS as the gold-standard baseline.

### Learning Goals
- Review GEMM math and memory footprint.
- Practice writing tiled CUDA kernels with shared memory.
- Compare naive, tiled, and cuBLAS implementations (cuBLAS hooks provided).
- Collect and interpret performance/roofline data.

### Directory Layout
- `src/main.cu` – entry point (argument parsing, host orchestration, cuBLAS integration).
- `src/gemm_kernel.cuh` – CUDA kernels and launch helpers (students fill in TODOs).
- `scripts/measure.sh` – example automation for sweeping matrix sizes.
- `data/` – placeholder for logs/results (created by the student).

### Getting Started
1. Install CUDA 12+ and ensure `nvcc` & `nvidia-smi` work.
2. Create/activate a conda or module environment if required by your cluster.
3. Run `make` (provided below) to build the starter binary `bin/dgemm`.
4. Execute `./bin/dgemm --m 2048 --n 2048 --k 2048 --impl baseline` to confirm the harness works before editing.

### Tasks
1. **Matrix math refresher**
   - Derive the FLOP count `2*M*N*K` and memory footprint. (No need to submit)
2. **Baseline host implementation (`src/main.cu`)**
   - Fill in the TODO blocks that allocate device buffers, copy data, launch kernels, and gather timing.
3. **CUDA kernels (`src/gemm_kernel.cuh`)**
   - Implement the naive element-per-thread kernel.
   - Implement the tiled shared-memory kernel (BLOCK_SIZE = 32 suggested).
   - Parameterize the launch configuration for arbitrary `M,N,K`.
4. **Performance study**
   - Use `scripts/measure.sh` as a template to sweep problem sizes.
   - Compare against cuBLAS (already linked). Report GFLOP/s gap and hypothesize causes.
   - Compare the performace across different tiling sizes (you may try 16, 32, 64, etc.).
5. **Report**
   - Include methodology, plots, speedups, and an analysis of memory vs compute limits.

### Deliverables
- Completed source with your TODOs resolved.
- A PDF report (≤5 pages) describing implementation details and performance.
- Raw timing logs or CSV files in `data/`.

### Rubric (20 pts)
- Correctness (6) – numerical accuracy vs cuBLAS reference implementation.
- Performance (6) – meets >30% of cuBLAS GFLOP/s for target sizes. You should achieve clear acceleration using tiled implementation compared to the naive implementation. (Aim for your best, you are able to achieve up to 70% of the cuBLAS FLOPs but it is not required)
- Analysis (4) – thoughtful discussion of bottlenecks.
- Presentation (4) – report clarity and code readability.

### Suggested Timeline
| Day | Milestone |
|-----|-----------|
| 1   | Build baseline, answer pre-lab |
| 2   | Implement naive kernel & validate correctness |
| 3   | Implement tiled kernel & optimize occupancy |
| 4   | Collect measurements, create plots |
| 5   | Finalize report & cleanup repo |

### Make Targets
Change the compute capacity in the Makefile to match your GPU version. (`sm_80` for A100, `sm_70` for V100)
```bash
make        # build bin/dgemm
make clean  # remove build artifacts
```

### Academic Integrity
You may discuss high-level ideas with classmates, but all code/report content must be your own. Cite any external references used.

