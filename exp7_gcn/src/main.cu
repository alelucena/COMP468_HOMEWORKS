#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gcn_layers.cuh"

struct Options {
    std::string graph_prefix = "data/cora";  // expects graph_prefix.csr, graph_prefix.feat, graph_prefix.label
    int hidden_dim = 128;
    int layers = 2;
    std::string impl = "baseline";  // baseline | fused
    bool verify = true;
    std::string dump_path = "";
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--graph") == 0 || strcmp(argv[i], "-g") == 0) && i + 1 < argc) {
            opt.graph_prefix = argv[++i];
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            opt.hidden_dim = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            opt.layers = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            opt.dump_path = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dgcn --graph data/cora --hidden 128 --layers 2 --impl baseline \\\n  [--dump outputs.bin] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    if (opt.hidden_dim <= 0 || opt.layers < 1) {
        throw std::invalid_argument("hidden and layers must be positive");
    }
    return opt;
}





int main(int argc, char** argv) {
    srand(42); // Set a fixed seed so rand() always produces the same sequence
    Options opt = parse_args(argc, argv);

    GraphData graph;
    /* TODO(student): load CSR graph + features + labels from opt.graph_prefix using helpers. */
    build_graph_from_files(opt.graph_prefix, graph);

    cusparseHandle_t cusparse;
    check_cusparse(cusparseCreate(&cusparse), "cusparseCreate");
    cublasHandle_t cublas;
    check_cublas(cublasCreate(&cublas), "cublasCreate");

    // Set up stream
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create stream");
    cusparseSetStream(cusparse, stream);
    cublasSetStream(cublas, stream);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    DeviceGCNWorkspace workspace;
    /* TODO(student): allocate device buffers for features, normalized adjacency, intermediate activations, weights. */
    allocate_device_graph(graph, workspace, opt.hidden_dim);

    // 1. Initialize weights on host
    size_t weights_count = (graph.feature_dim * opt.hidden_dim) + (opt.hidden_dim * graph.num_classes);
    std::vector<float> h_weights(weights_count);
    for (size_t i = 0; i < weights_count; ++i) {
        h_weights[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // 2. Dump weights so Python can use the SAME ones for verification
    if (opt.verify) {
        std::ofstream ofs("weights.bin", std::ios::binary);
        if (ofs) {
            ofs.write(reinterpret_cast<const char*>(h_weights.data()), h_weights.size() * sizeof(float));
            ofs.close();
            std::cout << "Weights dumped to weights.bin for verification." << std::endl;
        } else {
            std::cerr << "Error: Could not write to weights.bin" << std::endl;
        }
    }

    // 3. Move them to GPU
    check_cuda(cudaMemcpy(workspace.d_weights, h_weights.data(), 
                        weights_count * sizeof(float), 
                        cudaMemcpyHostToDevice), "Copy weights");


    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline") {
        check_cuda(cudaEventRecord(start), "record baseline start");
        /* TODO(student): run forward pass using cusparseSpMM + cublasSgemm per layer. */

        // Weights: W0 is [input_dim x hidden_dim], W1 is [hidden_dim x num_classes]
        float* d_W0 = workspace.d_weights;
        float* d_W1 = workspace.d_weights + (graph.feature_dim * opt.hidden_dim);

        // --- Layer 1 --
        // 1. Aggreation: temp = A_hat * features_in
        // A[ N x N] * X[N x feature_dim] -> [N x feature_dim]
        // rows=N, cols=feature_dim, K=N
        run_sparse_dense_mm(cusparse, workspace, 
                            graph.num_nodes, graph.feature_dim, graph.num_nodes,
                            workspace.d_features_in, workspace.d_temp);

        // 2. Weight multiply: features_out = temp * W0
        // [N x feature_dim] * [feature_dim x hidden_dim] -> [N * hidden_dim]
        // Signature: M, K, N (M=Nodes, K=Feat, N=Hidden)
        run_dense_layer(cublas,
                        graph.num_nodes, graph.feature_dim, opt.hidden_dim,
                        workspace.d_temp, d_W0, workspace.d_features_out);
        

        // 3. Activation
        apply_activation(workspace.d_features_out, graph.num_nodes * opt.hidden_dim, stream);

        // -- Layer 2 --

        // 4. Aggregation: temp = A_hat * features_out
        // [N x N] * [N * hidden_dim] -> [N * hidden_dim]
        // Signature: rows, cols, K
        run_sparse_dense_mm(cusparse, workspace,
                            graph.num_nodes, opt.hidden_dim, graph.num_nodes,
                            workspace.d_features_out, workspace.d_temp);

        // 5. Weight multiplication
        // [N * hidden_dim] * [hidden_dim x num_classes] -> [N x num_classes]
        // Signature: M, K, N (M=Nodes, K=Hidden, N=Classes)
        run_dense_layer(cublas, 
                        graph.num_nodes, opt.hidden_dim, graph.num_classes,
                        workspace.d_temp, d_W1, workspace.d_logits);
        


        check_cuda(cudaEventRecord(stop), "record baseline stop");
        check_cuda(cudaEventSynchronize(stop), "sync baseline stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed baseline"); 
    } else if (opt.impl == "fused") {
        check_cuda(cudaEventRecord(start), "record fused start");
        /* TODO(student): implement fused kernels (e.g., combine aggregation + activation) and time here. */

        // // Pointers to weight matrices
        float* d_W0 = workspace.d_weights;
        float* d_W1 = workspace.d_weights + (graph.feature_dim * opt.hidden_dim);

        // --- LAYER 1 ---
        // 1. Weight Multiply First: temp = features_in * W0
        // Do this first because it reduces the feature dimension early, 
        // making the custom sparse kernel much faster.
        run_dense_layer(cublas, graph.num_nodes, graph.feature_dim, opt.hidden_dim,
                        workspace.d_features_in, d_W0, workspace.d_temp);

        // 2. Fused Aggregation + ReLU: features_out = ReLU(A_hat * temp)
        dim3 block(16, 16);
        dim3 grid((graph.num_nodes + block.x - 1) / block.x, 
                  (opt.hidden_dim + block.y - 1) / block.y);
 
        fused_aggregation_act_kernel<<<grid, block, 0, stream>>>(
            workspace.d_csr_row_offsets, workspace.d_csr_col_indices, workspace.d_csr_values,
            workspace.d_temp, workspace.d_features_out,
            graph.num_nodes, opt.hidden_dim, true
        );

        // --- LAYER 2 ---
        // 3. Weight Multiply: temp = features_out * W1
        run_dense_layer(cublas, graph.num_nodes, opt.hidden_dim, graph.num_classes,
                        workspace.d_features_out, d_W1, workspace.d_temp);

        // 4. Fused Aggregation (No ReLU on last layer): logits = A_hat * temp
        dim3 grid_logits((graph.num_nodes + block.x - 1) / block.x, 
                         (graph.num_classes + block.y - 1) / block.y);

        fused_aggregation_act_kernel<<<grid_logits, block, 0, stream>>>(
            workspace.d_csr_row_offsets, workspace.d_csr_col_indices, workspace.d_csr_values,
            workspace.d_temp, workspace.d_logits,
            graph.num_nodes, graph.num_classes, false
        );

        check_cuda(cudaEventRecord(stop), "record fused stop");
        check_cuda(cudaEventSynchronize(stop), "sync fused stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
    } else {
        throw std::invalid_argument("Unknown --impl=" + opt.impl);
    }

    std::vector<float> h_logits(graph.num_nodes * graph.num_classes, 0.0f);
    /* TODO(student): copy device logits back into h_logits. */
    cudaMemcpyAsync(h_logits.data(), workspace.d_logits, graph.num_nodes * graph.num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (!opt.dump_path.empty()) {
        std::ofstream ofs(opt.dump_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
        }
        ofs.write(reinterpret_cast<const char*>(h_logits.data()),
                  static_cast<std::streamsize>(h_logits.size() * sizeof(float)));
        ofs.close();
    }

    if (opt.verify) {
        /* TODO(student): run DGL/PyTorch reference (e.g., via subprocess) or CPU path to compare logits. */
        std::string cmd = "python scripts/compare_with_dgl.py "
                  "--graph " + std::string(opt.graph_prefix) + "_dgl "
                  "--hidden " + std::to_string(opt.hidden_dim) + " "
                  "--layers " + std::to_string(opt.layers) + " "
                  "--outputs outputs.bin";

std::cout << "Running verification: " << cmd << std::endl;
std::system(cmd.c_str());
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2)
                  << "Impl=" << opt.impl
                  << " Graph=" << opt.graph_prefix
                  << " Hidden=" << opt.hidden_dim
                  << " Layers=" << opt.layers
                  << " Time(ms)=" << elapsed_ms
                  << " Edges/s=" << graph.nnz / (elapsed_ms * 1e-3)
                  << std::endl;
    } else {
        std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
    }

    /* TODO(student): free device buffers, destroy cuBLAS/cuSPARSE handles, destroy events. */
    
    // 1. Library Cleanup
    check_cusparse(cusparseDestroy(cusparse), "destroy cusparse");
    check_cublas(cublasDestroy(cublas), "destroy cublas");

    // 2. Graph
    destroy_device_graph(workspace);

    // 3. Synchronization/Event Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return 0;
}
