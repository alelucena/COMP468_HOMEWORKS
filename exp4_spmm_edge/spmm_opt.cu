
// spmm_opt.cu — STUDENT OPTIMIZATION SKELETON
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using float_t = float;

extern void load_csr_from_edgelist(const std::string& filename,
                                   int& M, int& K,
                                   std::vector<int>& row_ptr,
                                   std::vector<int>& col_idx,
                                   std::vector<float>& vals);

extern void spmm_cpu(int M, int K, int N,
                     const std::vector<int>& row_ptr,
                     const std::vector<int>& col_idx,
                     const std::vector<float>& vals,
                     const std::vector<float>& B,
                     std::vector<float>& C);

extern float max_abs_err(const std::vector<float>& A, const std::vector<float>& B);                       
/*
=================================================================
 OPT KERNEL — STUDENT TODO
=================================================================
*/
__global__ void spmm_csr_warp_kernel(
    int M, int N,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float_t* __restrict__ d_vals,
    const float_t* __restrict__ d_B,
    float_t* __restrict__ d_C)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = global_tid / 32;
    int lane = threadIdx.x % 32;

    if (warp >= M) return;
    int row = warp;

    // TODO student: fetch start, end
    // TODO student: for j = lane; j < N; j += 32 ...
    // TODO student: loop over nonzeros
    // TODO student: accumulate
}

int main() {
    int M, K;
    int N = 64;
    std::vector<int> row_ptr, col_idx;
    std::vector<float> vals;

    load_csr_from_edgelist("graph_edges.txt", M, K, row_ptr, col_idx, vals);
    int nnz = row_ptr.back();

    std::cout<<"Loaded matrix for optimized kernel test: M="<<M<<" K="<<K
             <<" nnz="<<row_ptr.back()<<"\n";
    
    std::vector<float> B((size_t)K * N);
    for (size_t i = 0; i < B.size(); i++) B[i] = float(rand()) / RAND_MAX;


    std::vector<float> C_ref;
    spmm_cpu(M, K, N, row_ptr, col_idx, vals, B, C_ref);


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


    int block = 256;
    long long total_threads = (long long)M * 32;
    int grid = (total_threads + block - 1) / block;

    std::cout << "Launching Warp Kernel: Grid=" << grid << ", Block=" << block << "\n";

    spmm_csr_warp_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B, d_C);
    cudaDeviceSynchronize();


    std::vector<float> C((size_t)M*N);
    cudaMemcpy(C.data(), d_C, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_err(C_ref, C);
    std::cout<<"Max error = "<<err<<"\n";
    if (err < 1e-5) std::cout << "Test PASSED" << std::endl;

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_vals);
    cudaFree(d_B); cudaFree(d_C);

    return 0;
}
