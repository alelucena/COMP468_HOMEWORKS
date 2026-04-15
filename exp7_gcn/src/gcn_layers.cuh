#pragma once

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include <fstream> // Required for std::ifstream
#include <algorithm> // Required for std::max
#include <stdexcept> // Required for std::runtime_error
#include <cmath> // Required for std::sqrt


struct GraphData {
    int num_nodes = 0;
    int num_edges = 0;  // undirected edges counted twice
    int nnz = 0;        // CSR nnz (including self loops)
    int feature_dim = 0;
    int num_classes = 0;

    std::vector<int> h_csr_row_offsets;
    std::vector<int> h_csr_col_indices;
    std::vector<float> h_csr_values;
    std::vector<float> h_features;   // row-major: num_nodes x feature_dim
    std::vector<int> h_labels;       // node labels
};

struct DeviceGCNWorkspace {
    int* d_csr_row_offsets = nullptr;
    int* d_csr_col_indices = nullptr;
    float* d_csr_values = nullptr;

    float* d_features_in = nullptr;
    float* d_features_out = nullptr;
    float* d_weights = nullptr;
    float* d_logits = nullptr;
    float* d_temp = nullptr;

    cusparseSpMatDescr_t spmat = nullptr;
};

inline void check_cusparse(cusparseStatus_t status, const char* msg) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuSPARSE error");
    }
}

inline void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

inline void build_graph_from_files(const std::string& prefix, GraphData& graph) {
    /* TODO(student):
       1. Read CSR metadata from prefix + ".csr" (e.g., row ptr count, nnz, columns, values).
       2. Read dense features from prefix + ".feat" (float32 rows).
       3. Read labels from prefix + ".label" (int32).
       4. Populate graph struct and add self-loops + normalization coefficients. */

    // --- 1: LOAD GRAPH (CSR) ---
    // The .csr file contains the adjacency structure.
    // Format: [int32 num_nodes][int32 nnz][row_offsets...][col_indices...]
    std::ifstream ifs(prefix + ".csr", std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open " + prefix + ".csr");

    int32_t num_nodes, nnz;
    ifs.read(reinterpret_cast<char*>(&num_nodes), sizeof(int32_t));
    ifs.read(reinterpret_cast<char*>(&nnz), sizeof(int32_t));

    graph.num_nodes = num_nodes;
    graph.nnz = nnz;

    // Allocate host memory for CSR arrays
    graph.h_csr_row_offsets.resize(num_nodes + 1);
    graph.h_csr_col_indices.resize(nnz);
    graph.h_csr_values.resize(nnz); 

    // row_offsets[i] is where the neighbor list for node 'i' starts in col_indices
    ifs.read(reinterpret_cast<char*>(graph.h_csr_row_offsets.data()), (num_nodes + 1) * sizeof(int32_t));
    // col_indices contains the actual IDs of the target neighbors
    ifs.read(reinterpret_cast<char*>(graph.h_csr_col_indices.data()), nnz * sizeof(int32_t));
    ifs.close();


    // --- 2: LOAD NODE FEATURES ---
    // The .feat file contains a raw float32 array of shape [num_nodes, feature_dim].
    // Since the Python script doesn't write a header, I calculate feature_dim from file size.
    std::ifstream feat_file(prefix + ".feat", std::ios::binary);
    if (!feat_file) throw std::runtime_error("Could not open " + prefix + ".feat");

    feat_file.seekg(0, std::ios::end);
    size_t feat_file_size = feat_file.tellg();
    feat_file.seekg(0, std::ios::beg);
    
    // total_bytes = num_nodes * feature_dim * sizeof(float)
    graph.feature_dim = feat_file_size / (graph.num_nodes * sizeof(float));

    size_t total_features = static_cast<size_t>(graph.num_nodes) * graph.feature_dim;
    graph.h_features.resize(total_features);
    feat_file.read(reinterpret_cast<char*>(graph.h_features.data()), total_features * sizeof(float));
    feat_file.close();


    // --- 3: LOAD LABELS ---
    // The .label file contains an int32 array of shape [num_nodes].
    std::ifstream label_file(prefix + ".label", std::ios::binary);
    if (!label_file) throw std::runtime_error("Could not open " + prefix + ".label");

    graph.h_labels.resize(graph.num_nodes);
    label_file.read(reinterpret_cast<char*>(graph.h_labels.data()), graph.num_nodes * sizeof(int32_t));
    
    // Infer the number of classes by finding the highest label index
    int max_label = 0;
    for(int l : graph.h_labels) if(l > max_label) max_label = l;
    graph.num_classes = max_label + 1;
    label_file.close();


    // --- 4: SYMMETRIC NORMALIZATION ---
    // Compute A_hat = D^-0.5 * (A + I) * D^-0.5
    // Python script already added self-loops (the +I part), so I only compute coefficients.

    // Initialize all edge weights to 1.0 before normalization
    graph.h_csr_values.assign(graph.nnz, 1.0f);

    // Compute the degree (d_i) for each node.
    // In GCNs, degree = sum of weights in the row (including the self-loop).
    std::vector<float> degrees(graph.num_nodes, 0.0f);
    for (int i = 0; i < graph.num_nodes; ++i) {
        int start = graph.h_csr_row_offsets[i];
        int end = graph.h_csr_row_offsets[i + 1];
        for (int j = start; j < end; ++j) {
            degrees[i] += graph.h_csr_values[j]; 
        }
    }

    // Replace 1.0 weights with the symmetric normalization coefficient:
    // weight(i,j) = 1 / sqrt(degree(i) * degree(j))
    // This scales the message passed from node 'j' to node 'i' by their relative connectivity.
    for (int i = 0; i < graph.num_nodes; ++i) {
        int start = graph.h_csr_row_offsets[i];
        int end = graph.h_csr_row_offsets[i + 1];
        for (int j = start; j < end; ++j) {
            int neighbor_idx = graph.h_csr_col_indices[j];
            
            // Note: degrees[i] and degrees[neighbor_idx] are >= 1 due to self-loops.
            float norm_coeff = 1.0f / std::sqrt(degrees[i] * degrees[neighbor_idx]);
            graph.h_csr_values[j] = norm_coeff;
        }
    }
}

inline void allocate_device_graph(const GraphData& graph, DeviceGCNWorkspace& workspace, int hidden_dim) {
    /* TODO(student): cudaMalloc / cudaMemcpy CSR + feature buffers, create cusparse descriptors. */

    // 1. Allocate CSR Graph Structure
    check_cuda(cudaMalloc(&workspace.d_csr_row_offsets, (graph.num_nodes + 1) * sizeof(int)), "malloc row_offsets");
    check_cuda(cudaMalloc(&workspace.d_csr_col_indices, graph.nnz * sizeof(int)), "malloc col_indices");
    check_cuda(cudaMalloc(&workspace.d_csr_values, graph.nnz * sizeof(float)), "malloc csr_values");

    // 2. Allocate Feature and Intermediate Buffers
    // Input Features: [N x feature_dim]
    check_cuda(cudaMalloc(&workspace.d_features_in, graph.num_nodes * graph.feature_dim * sizeof(float)), "malloc features_in");
    
    // Hidden Layer Out: [N x hidden_dim] 
    check_cuda(cudaMalloc(&workspace.d_features_out, graph.num_nodes * hidden_dim * sizeof(float)), "malloc features_out");

    // Weights: [feat_dim * hidden_dim] + [hidden_dim * num_classes]
    size_t weights_size = (graph.feature_dim * hidden_dim) + (hidden_dim * graph.num_classes);
    check_cuda(cudaMalloc(&workspace.d_weights, weights_size * sizeof(float)), "malloc weights");

    // Logits: [N x num_classes]
    check_cuda(cudaMalloc(&workspace.d_logits, graph.num_nodes * graph.num_classes * sizeof(float)), "malloc logits");

    // Temp Buffer. Must be large enough for [N x max(feat_dim, hidden_dim)]
    size_t temp_cols = std::max(graph.feature_dim, hidden_dim);
    check_cuda(cudaMalloc(&workspace.d_temp, graph.num_nodes * temp_cols * sizeof(float)), "malloc temp");

    // 3. Copy Data from Host to Device
    check_cuda(cudaMemcpy(workspace.d_csr_row_offsets, graph.h_csr_row_offsets.data(), (graph.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice), "memcpy row_offsets");
    check_cuda(cudaMemcpy(workspace.d_csr_col_indices, graph.h_csr_col_indices.data(), graph.nnz * sizeof(int), cudaMemcpyHostToDevice), "memcpy col_indices");
    check_cuda(cudaMemcpy(workspace.d_csr_values, graph.h_csr_values.data(), graph.nnz * sizeof(float), cudaMemcpyHostToDevice), "memcpy csr_values");
    check_cuda(cudaMemcpy(workspace.d_features_in, graph.h_features.data(), graph.num_nodes * graph.feature_dim * sizeof(float), cudaMemcpyHostToDevice), "memcpy features");

    // 4. Create cuSPARSE CSR Descriptor
    check_cusparse(cusparseCreateCsr(&workspace.spmat, 
                                     graph.num_nodes, graph.num_nodes, graph.nnz,
                                     workspace.d_csr_row_offsets, 
                                     workspace.d_csr_col_indices, 
                                     workspace.d_csr_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, // Index types
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),  // Base and Value type
                   "create csr descriptor");
}

inline void destroy_device_graph(DeviceGCNWorkspace& workspace) {
    /* TODO(student): destroy descriptors and cudaFree buffers. */

    // 1. Destroy Sparse Descriptor
    if (workspace.spmat) {
        check_cusparse(cusparseDestroySpMat(workspace.spmat), "destroy spmat");
        workspace.spmat = nullptr;
    }

    // 2. Free Device Buffers
    cudaFree(workspace.d_csr_row_offsets);
    cudaFree(workspace.d_csr_col_indices); 
    cudaFree(workspace.d_csr_values);
    cudaFree(workspace.d_features_in);
    cudaFree(workspace.d_features_out);
    cudaFree(workspace.d_weights);
    cudaFree(workspace.d_logits);
    cudaFree(workspace.d_temp);

    // 3. Nullify pointers to prevent dangling references
    workspace.d_csr_row_offsets = nullptr;
    workspace.d_csr_col_indices = nullptr;
    workspace.d_csr_values = nullptr;
    workspace.d_features_in = nullptr;
    workspace.d_features_out = nullptr;
    workspace.d_weights = nullptr;
    workspace.d_logits = nullptr;
    workspace.d_temp = nullptr;

}

inline void run_sparse_dense_mm(cusparseHandle_t handle,
                                DeviceGCNWorkspace& workspace,
                                int rows,
                                int cols,
                                int K,
                                const float* d_input,
                                float* d_output) {
    /* TODO(student): configure cusparseDnMatDescr_t for input/output and call cusparseSpMM.
       rows = num_nodes, cols = hidden_dim, K = feature_dim. */

    // Initializations
    cusparseDnMatDescr_t dn_in, dn_out;
    float alpha = 1.0f;
    float beta  = 0.0f;
    
    // 1. Create descriptor for input:     
    // GCN is in row-major layout: [K x cols], so the leading dimension is cols
    check_cusparse(cusparseCreateDnMat(
        &dn_in,
        K, cols, cols,
        const_cast<float*>(d_input),
        CUDA_R_32F, CUSPARSE_ORDER_ROW
    ), "create d_in");

    // 2. Create descriptor for output
    // Row major: Layout is [rows x cols], so the leading dimension is cols'
     check_cusparse(cusparseCreateDnMat(
        &dn_out,
        rows, cols, cols,
        const_cast<float*>(d_output),
        CUDA_R_32F, CUSPARSE_ORDER_ROW
    ), "create d_in");

    // 3. Determine workspace size for SpMM
    size_t workspaceSize = 0;
    check_cusparse(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, // Op A (sparse)
        CUSPARSE_OPERATION_NON_TRANSPOSE, // Op B (dense)
        &alpha, workspace.spmat, dn_in, &beta, dn_out,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &workspaceSize
    ), "SpMM_bufferSize");

    // 4. Allocate temporary buffer if needed
    void * d_spmm_work = nullptr;
    if (workspaceSize > 0) {
        check_cuda(cudaMalloc(&d_spmm_work, workspaceSize), "malloc SpMM work");
    }

    // 5. Execute SpMM: C = alpha * A * B  + beta * C
    check_cusparse(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, workspace.spmat, dn_in, &beta, dn_out,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_work
    ), "SpMM execute");

    // 6. Clean up local descriptors and temporary buffer
    if (d_spmm_work) cudaFree(d_spmm_work);
    check_cusparse(cusparseDestroyDnMat(dn_in), "destroy dn_in");
    check_cusparse(cusparseDestroyDnMat(dn_out), "destroy dn_out");

}

inline void run_dense_layer(cublasHandle_t handle,
                            int M,
                            int K,
                            int N,
                            const float* d_input,
                            const float* d_weight,
                            float* d_output) {
    /* TODO(student): call cublasSgemm to compute (M x K) * (K x N). */
    float alpha = 1.0f;
    float beta = 0.0f;

    // To get Row-Major C = A * B, we compute Col-Major C = (B_T * A_T)_T
    // We pass B as the first argument and A as the second.
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K,           // Swap of N and M
                &alpha, 
                d_weight, N,       // Weight matrix (B)
                d_input, K,        // Input matrix (A)
                &beta, 
                d_output, N);      // Output matrix (C)
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] > 0.0f) ? data[idx] : 0.0f;
    }
}

inline void apply_activation(float* d_tensor, int elements, cudaStream_t stream) {
    /* TODO(student): implement ReLU or ELU kernel. */

    int threads = 256;
    int blocks = (elements + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(d_tensor, elements);
}

inline void apply_dropout(float* d_tensor, int elements, float drop_prob, cudaStream_t stream) {
    /* TODO(student): optional – implement dropout. */
    (void)d_tensor;
    (void)elements;
    (void)drop_prob;
    (void)stream;
}

inline void softmax_cross_entropy(const float* d_logits,
                                  const int* d_labels,
                                  int num_nodes,
                                  int num_classes,
                                  float* d_loss) {
    /* TODO(student): compute loss/accuracy or copy logits for host-side evaluation. */
    (void)d_logits;
    (void)d_labels;
    (void)num_nodes;
    (void)num_classes;
    (void)d_loss;
}

__global__ void fused_linear_relu_kernel(
    const float* X, // [M x K] Row-Major
    const float* W, // [K x N] Row-Major
    float* Y,       // [M x N] Row-Major
    int M, int K, int N) 
{
    // row corresponds to nodes (M), col corresponds to output features (N)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // Both are indexed as Row-Major
            acc += X[row * K + k] * W[k * N + col];
        }

        // Apply ReLU (Activation Fusion)
        Y[row * N + col] = (acc > 0.0f) ? acc : 0.0f;
    }
}