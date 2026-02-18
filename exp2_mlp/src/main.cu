#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlp_layers.cuh"

struct Options {
    std::vector<int> layers = {1024, 2048, 1024};  // includes input dim and final output dim
    int batch = 128;
    std::string activation = "relu";
    std::string impl = "baseline";  // baseline | activation_fused
    bool verify = true;
};

std::vector<int> parse_layers_list(const std::string& csv) {
    std::vector<int> dims;
    size_t start = 0;
    while (start < csv.size()) {
        size_t comma = csv.find(',', start);
        const size_t len = (comma == std::string::npos) ? (csv.size() - start) : (comma - start);
        if (len > 0) {
            dims.push_back(std::stoi(csv.substr(start, len)));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return dims;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            opt.layers = parse_layers_list(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            opt.batch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) {
            opt.activation = argv[++i];
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dmlp --layers 1024,2048,1024 --batch 128 --activation relu \\\n  --impl baseline|activation_fused [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    if (opt.layers.size() < 2) {
        throw std::invalid_argument("--layers must contain at least two integers (input/output)");
    }
    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

void seed_tensor(std::vector<float>& data, float scale) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = scale * std::sin(0.11f * static_cast<float>(i));
    }
}

void mlp_cpu_reference(const std::vector<int>& layers,
                       int batch,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases,
                       const std::vector<size_t>& weight_offsets,
                       const std::vector<size_t>& bias_offsets,
                       const std::vector<float>& input,
                       std::vector<float>& output,
                       const std::string& activation) {
    /* TODO(student): implement a simple CPU forward pass (GEMM + bias + activation per layer).
       Remember that weights are stored row-major with shape [out_dim, in_dim]. */

    // 1. Init copy_input with the initial input data.
    std::vector<float> copy_input = input;

    // Iterate through layers:
    for (size_t layer = 0; layer < layers.size() - 1; ++layer) {
        // Layer transitions: 0 -> 1, 1 -> 2
        int in_dim = layers[layer]; // row width
        int out_dim = layers[layer + 1]; // col length

         // Init buffer for layer results.
        std::vector<float> result_input(batch * out_dim);

        // 2. Perform batched GEMM
        // weight shape: [out_dim, in_dim]
        // input shape: [batch, in_dim]
        // output shape: [batch, out_dim]

        for (int num = 0; num < batch; ++num) {

            // j-th output neuron (row)
            for (int j = 0; j < out_dim; ++j) {
                float c_sum = 0.0f;

                // i-th input neuron (col)
                for (int i = 0; i < in_dim; ++i) {
                    // GEMM call: weight[j, i] + input[num, i]
                    float w = weights[weight_offsets[layer] + j * in_dim + i];
                    float x = copy_input[num * in_dim + i];
                    c_sum +=  w * x;
                }

                // 3. Bias (same bias added to every sample in the batch)
                // Bias shape: [out_dim]
                c_sum += biases[bias_offsets[layer] + j];

                // Apply Relu activation
                if (activation == "relu") {
                    c_sum = std::max(0.0f, c_sum);
                }

                result_input[num * out_dim + j] = c_sum;
            }
        }

        // Continue to the next layer
        copy_input = std::move(result_input);

    }

    // 5. Copy final result to output.
    output = copy_input;
   
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    const int batch = opt.batch;
    const size_t input_elems = static_cast<size_t>(batch) * opt.layers.front();
    const size_t output_elems = static_cast<size_t>(batch) * opt.layers.back();
    const int num_layers = static_cast<int>(opt.layers.size()) - 1;

    std::vector<size_t> weight_offsets(num_layers, 0);
    std::vector<size_t> bias_offsets(num_layers, 0);
    size_t weight_cursor = 0;
    size_t bias_cursor = 0;
    for (int i = 0; i < num_layers; ++i) {
        const int in_dim = opt.layers[i];
        const int out_dim = opt.layers[i + 1];
        weight_offsets[i] = weight_cursor;
        bias_offsets[i] = bias_cursor;
        weight_cursor += static_cast<size_t>(out_dim) * in_dim;
        bias_cursor += static_cast<size_t>(out_dim);
    }




    std::vector<float> h_input(input_elems);
    std::vector<float> h_weights(weight_cursor);
    std::vector<float> h_biases(bias_cursor);
    std::vector<float> h_output(output_elems, 0.0f);
    std::vector<float> h_ref(output_elems, 0.0f);

    seed_tensor(h_input, 1.0f);
    seed_tensor(h_weights, 0.25f);
    seed_tensor(h_biases, 0.01f);

    float* d_input = nullptr;
    float* d_workspace_a = nullptr;
    float* d_workspace_b = nullptr;
    float* d_weights = nullptr;
    float* d_biases = nullptr;
    /* TODO(student): allocate device buffers (activations + weights + biases) and copy host data. */


    // Allocate device buffers
    cudaMalloc(&d_input, input_elems * sizeof(float));
    cudaMalloc(&d_weights, weight_cursor * sizeof(float));
    cudaMalloc(&d_biases, bias_cursor * sizeof(float));

    // Allocate two workspaces for back-and-forth. 
    // Find max width needed for any hidden or output layer
    size_t max_num_layers = 0;
    for (size_t i = 1; i < opt.layers.size(); ++i) {
        max_num_layers = std::max(max_num_layers, (size_t)batch * opt.layers[i]);
    }
    cudaMalloc(&d_workspace_a, max_num_layers * sizeof(float));
    cudaMalloc(&d_workspace_b, max_num_layers * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_input, h_input.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), weight_cursor * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases.data(), bias_cursor * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create stream");

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    // Copy d_input into d_workspace_a
    cudaMemcpyAsync(d_workspace_a, d_input, input_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline") {
        check_cuda(cudaEventRecord(start, stream), "record baseline start");
        for (int layer = 0; layer < num_layers; ++layer) {
            LayerShape shape{batch, opt.layers[layer], opt.layers[layer + 1]};
            const float* d_w = d_weights + weight_offsets[layer];  // TODO(student): offset into d_weights based on layer. Pointer arithmetic to jump to the correct memory address.
            const float* d_b = d_biases + bias_offsets[layer];  // TODO(student): offset into d_biases based on layer. Pointer arithmetic to jump to the correct memory address..
            run_gemm_layer(d_workspace_a, d_w, d_workspace_b, shape, handle);
            launch_bias_add(d_b, d_workspace_b, shape, stream);
            launch_activation(opt.activation, d_workspace_b, shape, stream);
            std::swap(d_workspace_a, d_workspace_b);
        }
        check_cuda(cudaEventRecord(stop, stream), "record baseline stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed baseline");
    } else if (opt.impl == "activation_fused") {
        check_cuda(cudaEventRecord(start, stream), "record fused start");
        for (int layer = 0; layer < num_layers; ++layer) {
            LayerShape shape{batch, opt.layers[layer], opt.layers[layer + 1]};
            const float* d_w = d_weights + weight_offsets[layer]; // TODO(student)
            const float* d_b = d_biases + bias_offsets[layer];   // TODO(student)
            run_gemm_layer(d_workspace_a, d_w, d_workspace_b, shape, handle);
            launch_fused_bias_activation(d_b, opt.activation, d_workspace_b, shape, stream);
            std::swap(d_workspace_a, d_workspace_b);
        }
        check_cuda(cudaEventRecord(stop, stream), "record fused stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
    } else {
        throw std::invalid_argument("Unknown --impl " + opt.impl);
    }

    /* TODO(student): copy final activations back to h_output. */
    cudaMemcpyAsync(h_output.data(), d_workspace_a, output_elems * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (opt.verify) {
        mlp_cpu_reference(opt.layers,
                          batch,
                          h_weights,
                          h_biases,
                          weight_offsets,
                          bias_offsets,
                          h_input,
                          h_ref,
                          opt.activation);
        /* TODO(student): compute max absolute difference between h_output and h_ref. */
        float max_diff = 0.0f;
        for (size_t i = 0; i < h_ref.size(); ++i) {
            float diff = std::abs(h_output[i] - h_ref[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        std::cout << "Max absolute difference: " << max_diff << std::endl;
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Impl=" << opt.impl << " Batch=" << batch << " Layers=";
        for (size_t i = 0; i < opt.layers.size(); ++i) {
            std::cout << opt.layers[i];
            if (i + 1 < opt.layers.size()) {
                std::cout << "x";
            }
        }
        std::cout << " Time(ms)=" << elapsed_ms
                  << " GFLOP/s=" << mlp_gflops(opt.layers, batch, elapsed_ms) << std::endl;
    } else {
        std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
    }

    /* TODO(student): cleanup (cudaFree buffers, destroy events/stream/handle). */

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_workspace_a);
    cudaFree(d_workspace_b);

    // Destroy cuBLAS handle
    if (handle) {
        cublasDestroy(handle);
    }

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Destroy stream
    if (stream) {
        cudaStreamDestroy(stream);
    }
    return 0;
}
