#pragma once

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;

inline dim3 make_grid(int m, int n) {
    return dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1);
}

__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    /* TODO(student): compute row/col indices, accumulate dot product, write to C */
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // Rows are calculated using the 'y' dimensio.
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary Check
    if (row < M && col < N) {
        //float value = 0;

        // POISON LINES:
        C[row * N + col] = 1234.56f; 
        return;

        // // Dot product
        // for (int k = 0; k < K; k++) {
        //     // A is M * K and B is K * N
        //     value += A[row * K + k] * B[k * N + col];
        // }

        // // Set value in C matrix
        // C[row * N + col] = value;
    }


}

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    /* TODO(student): use shared memory tiles of size BLOCK_SIZE x BLOCK_SIZE */
}

inline void launch_naive_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N);
    /* TODO(student): launch gemm_naive_kernel with provided stream. */

    // 3rd argument inside <<< >>> is the Shared Memory Size -> 0 because not using dynamic shared memory
    gemm_naive_kernel<<<grid, block, 0, stream >>>(d_a, d_b, d_c, M, N, K);
}

inline void launch_tiled_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    [[maybe_unused]] dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N);
    /* TODO(student): launch gemm_tiled_kernel and check for errors. */
    (void)d_a;
    (void)d_b;
    (void)d_c;
    (void)grid;
    (void)block;
    (void)stream;
}

