
// spmm_ref.cpp
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <cassert>

using float_t = float;

// generate random CSR matrix
void generate_random_csr(int M, int K, double density,
                         std::vector<int>& row_ptr,
                         std::vector<int>& col_idx,
                         std::vector<float_t>& vals,
                         unsigned seed=1234) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float_t> ud(0.0f, 1.0f);
    std::uniform_real_distribution<float_t> valdist(-1.0f, 1.0f);

    row_ptr.assign(M+1, 0);
    std::vector<std::vector<int>> cols(M);
    for (int i=0;i<M;i++){
        for (int j=0;j<K;j++){
            if (ud(rng) < density) cols[i].push_back(j);
        }
        if (cols[i].empty()) cols[i].push_back(i % K);
        row_ptr[i+1] = row_ptr[i] + (int)cols[i].size();
    }
    int nnz = row_ptr.back();
    col_idx.resize(nnz);
    vals.resize(nnz);
    int p=0;
    for (int i=0;i<M;i++){
        for (int c: cols[i]){
            col_idx[p] = c;
            vals[p] = valdist(rng);
            p++;
        }
    }
}

// CPU reference: C = A (CSR) * B (dense)
void spmm_cpu(int M, int K, int N,
              const std::vector<int>& row_ptr,
              const std::vector<int>& col_idx,
              const std::vector<float_t>& vals,
              const std::vector<float_t>& B, // K x N, row-major
              std::vector<float_t>& C) {    // M x N, row-major
    C.assign((size_t)M*N, 0.0f);
    for (int i=0;i<M;i++){
        for (int p = row_ptr[i]; p < row_ptr[i+1]; ++p){
            int k = col_idx[p];
            float_t v = vals[p];
            const float_t* Brow = &B[(size_t)k * N];
            float_t* Crow = &C[(size_t)i * N];
            for (int j=0;j<N;j++){
                Crow[j] += v * Brow[j];
            }
        }
    }
}

// small utility: max abs difference
float_t max_abs_err(const std::vector<float_t>& A, const std::vector<float_t>& B){
    assert(A.size()==B.size());
    float_t mx = 0;
    for (size_t i=0;i<A.size();++i){
        float_t d = std::abs(A[i]-B[i]);
        if (d>mx) mx=d;
    }
    return mx;
}
