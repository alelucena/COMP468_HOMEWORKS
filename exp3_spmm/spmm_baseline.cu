
// spmm_baseline.cu — STUDENT SKELETON
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include "spmm_ref.cpp"

using float_t = float;

/*
===============================================================
 BASELINE KERNEL — one thread processes ONE ROW of A
 STUDENT TODO: 
   - Fill missing loops
   - Compute C[row, j] += value * B[k, j]
===============================================================
*/
__global__ void spmm_csr_row_kernel(
    int M, int N,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float_t* __restrict__ d_vals,
    const float_t* __restrict__ d_B,
    float_t* __restrict__ d_C) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // TODO (student): Initialize output row C[row, :]
    for (int j = 0; j < N; j++) {
        d_C[(size_t)row * N + j] = 0.0f;
    }

    // Find nonzero range
    int start, end;
    // TODO (student): load start, end  (range logic)
    start = d_row_ptr[row];
    end = d_row_ptr[row + 1];

    // Loop over nonzeros in this row (iterates through the Value and Col arrays for the slice belonging to this thread's row)
    for (int i = start; i < end; i++)
    // TODO (student): 
    {
        // TODO (student): retrieve column index k 
        int k = d_col_idx[i]; // which row of matrix B to multiply by.
        // TODO (student): retrieve value v 
        float v = d_vals[i]; // the value from matrix A.

        // TODO (student): loop over all columns j of output (0..N-1)
        //                 and accumulate:
        for (int j = 0; j < N; j++) {
            // For every nnz value "v" at A[row, k], multiply it by the entire k-th row of B and add it to the rowth-th row of C.
            d_C[(size_t)row * N + j] += v * d_B[(size_t)k * N + j];
        }
       
    }
}

/*
===============================================================
 MAIN PROGRAM
===============================================================
*/
int main(int argc, char** argv) {
    int M = 512, K = 512, N = 64;
    double density = 0.01;
    if (argc > 1) density = std::atof(argv[1]);
    unsigned seed = 1234;

    std::vector<int> row_ptr, col_idx;
    std::vector<float_t> vals;
    generate_random_csr(M, K, density, row_ptr, col_idx, vals, seed);
    int nnz = row_ptr.back();
    std::cout << "nnz = " << nnz << "\\n";

    // Create B
    std::vector<float> B((size_t)K * N);
    for (size_t i = 0; i < B.size(); i++) B[i] = float(rand()) / RAND_MAX;


    // CPU reference
    std::vector<float> C_ref;
    spmm_cpu(M, K, N, row_ptr, col_idx, vals, B, C_ref);

    // Copy to device
    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_B, *d_C;
    cudaMalloc(&d_row_ptr, (M+1)*sizeof(int));
    cudaMalloc(&d_col_idx, nnz*sizeof(int));
    cudaMalloc(&d_vals, nnz*sizeof(float));
    cudaMalloc(&d_B, (size_t)K*N*sizeof(float));
    cudaMalloc(&d_C, (size_t)M*N*sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals.data(), nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), (size_t)K*N*sizeof(float), cudaMemcpyHostToDevice);

    // --- TIMING START ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch incomplete student kernel
    int block = 256;
    int grid = (M + block - 1) / block;

    // Warmup launch
    spmm_csr_row_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B, d_C);
    cudaDeviceSynchronize();    


    cudaEventRecord(start);
    spmm_csr_row_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // --- TIMING END ---

    // GFLOPS calculation: (2 * nnz * N) operations
    double seconds = milliseconds / 1000.0;
    double gflops = (2.0 * nnz * N) / (seconds * 1e9);

    std::cout << "Density: " << density << " | NNZ: " << nnz << "\n";
    std::cout << "Throughput: " << gflops << " GFLOPS" << "\n";

    // Copy back
    std::vector<float> C((size_t)M*N);
    cudaMemcpy(C.data(), d_C, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Compare (will be wrong until students complete TODOs)
    float err = max_abs_err(C_ref, C);
    std::cout << "Max error = " << err << "\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_vals);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
