
# SpMM CUDA Experiment — Edge List Version

This experiment covers Sparse Matrix-Matrix Multiplication (SpMM) on GPUs using the CSR format. Students implement two CUDA kernels: a baseline row-per-thread approach and an optimized warp-based kernel.

## Learning Goals
- Understand CSR sparse matrix representation.
- Implement SpMM kernels with different parallelization strategies.
- Compare row-per-thread vs warp-based approaches.

## Input format: `graph_edges.txt`
Each line:
```
u v
```
means A[u, v] = 1.

## Build
```
make
```

## Run
```
./spmm_baseline
./spmm_opt
```

## Tasks
1. **Baseline Kernel (`spmm_baseline.cu`)**
   - Implement the row-per-thread SpMM kernel where each thread computes one output row.
   - Fill in the TODO blocks to load row pointers, iterate over non-zeros, and accumulate results.

2. **Optimized Kernel (`spmm_opt.cu`)**
   - Implement the warp-based SpMM kernel where each warp handles one row.
   - Use warp-level parallelism to distribute work across the output columns.

3. **Validation**
   - Verify correctness against the CPU reference (max error < 1e-5).

## Deliverables
- Completed CUDA source files with TODOs resolved.
- A brief report (1-2 pages) including:
  - Explanation of your implementation approach.
  - Clear plots showing performance comparison (e.g., speedup of warp kernel vs baseline).
  - Discussion of any acceleration results observed.

## Rubric (20 pts)
- **Correctness (10)** – Both kernels produce results matching the CPU reference within tolerance.
- **Code Quality (5)** – Clear, readable code with appropriate comments.
- **Report (5)** – Clear explanation of implementation, plots showing performance/acceleration results.

## Academic Integrity
You may discuss ideas with classmates, but all code and report content must be your own.
