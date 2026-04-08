#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lenet_layers.cuh"

struct Options {
    int batch = 32;
    std::string algo = "implicit_gemm";       // cuDNN conv algo hint
    std::string impl = "baseline";            // baseline | fused
    bool verify = true;
    std::string dump_path = "";               // optional binary file for logits
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--batch") == 0 || strcmp(argv[i], "-b") == 0) && i + 1 < argc) {
            opt.batch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            opt.algo = argv[++i];
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            opt.dump_path = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dlenet --batch N --algo implicit_gemm --impl baseline|fused \\\n  [--dump outputs.bin] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    if (opt.batch <= 0) {
        throw std::invalid_argument("Batch must be > 0");
    }
    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void seed_tensor(std::vector<float>& vec, float scale) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = scale * std::sin(0.017f * static_cast<float>(i));
    }
}

void lenet_cpu_reference(const Options& opt,
                         const LenetShape& shape,
                         const std::vector<float>& weights,
                         const std::vector<size_t>& weight_offsets,
                         const std::vector<float>& biases,
                         const std::vector<size_t>& bias_offsets,
                         const std::vector<float>& input,
                         std::vector<float>& output) {
    /* TODO(student): implement a simple CPU LeNet forward (conv/pool/activations/GEMM).
       Keep it single-threaded for simplicity or call into a reference framework. */

    // --- STEP 1: CONVOLUTION 1 (1 -> 6 channels, 32x32 -> 28x28) ---
    int c1_dim = 28;
    std::vector<float> c1_out(shape.conv1_out_elems);
    for (int n = 0; n < opt.batch; ++n) {
        for (int oc = 0; oc < shape.conv1_out_channels; ++oc) {
            float b = biases[bias_offsets[0] + oc];
            for (int oh = 0; oh < c1_dim; ++oh) {
                for (int ow = 0; ow < c1_dim; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < shape.in_channels; ++ic) {
                        for (int kh = 0; kh < 5; kh++) {
                            for (int kw = 0; kw < 5; kw++) {
                                size_t i_idx = ((n * shape.in_channels + ic) * 32 + (oh + kh)) * 32 + (ow + kw);
                                size_t w_idx = weight_offsets[0] + ((oc * shape.in_channels + ic) * 25 + kh * 5 + kw);
                                sum += input[i_idx] * weights[w_idx];
                            }
                        }
                    }
                    // Apply Tanh activation
                    c1_out[((n * 6 + oc) * 28 + oh) * 28 + ow] = std::tanh(sum + b);
                }
            }
        }
    }

    // --- STEP 2: POOL 1 (Average Pool, 28x28 -> 14x14) ---
    int s2_dim = 14; 
    std::vector<float> s2_out(shape.pool1_out_elems);
    for (int n = 0; n < opt.batch; ++n) {
        for (int c = 0; c < 6; ++c) {
            for (int oh = 0; oh < s2_dim; ++oh) {
                for (int ow = 0; ow < s2_dim; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            sum += c1_out[((n * 6 + c) * 28 + (oh * 2 + kh)) * 28 + (ow * 2 + kw)];
                        }
                    }
                    s2_out[((n * 6 + c) * 14 + oh) * 14 + ow] = sum / 4.0f;
                }
            }
        }
    }

    // --- STEP 3: CONVOLUTION 2 (6 -> 16 channels, 14x14 -> 10x10) ---
    int c3_dim = 10;
    std::vector<float> c3_out(shape.conv2_out_elems);
    for (int n = 0; n < opt.batch; ++n) {
        for (int oc = 0; oc < 16; ++oc) {
            float b = biases[bias_offsets[1] + oc];
            for (int oh = 0; oh < c3_dim; ++oh) {
                for (int ow = 0; ow < c3_dim; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < 6; ++ic) {
                        for (int kh = 0; kh < 5; kh++) {
                            for (int kw = 0; kw < 5; kw++) {
                                size_t i_idx = ((n * 6 + ic) * 14 + (oh + kh)) * 14 + (ow + kw);
                                size_t w_idx = weight_offsets[1] + ((oc * 6 + ic) * 25 + kh * 5 + kw);
                                sum += s2_out[i_idx] * weights[w_idx];
                            }
                        }
                    }
                    // Apply Tanh activation
                    c3_out[((n * 16 + oc) * 10 + oh) * 10 + ow] = std::tanh(sum + b);
                }
            }
        }
    }

    // --- STEP 4: POOL 2 (Average Pool, 10x10 -> 5x5) ---
    int s4_dim = 5;
    std::vector<float> s4_out(shape.pool2_out_elems);
    for (int n = 0; n < opt.batch; ++n) {
        for (int c = 0; c < 16; ++c) {
            for (int oh = 0; oh < s4_dim; ++oh) {
                for (int ow = 0; ow < s4_dim; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            sum += c3_out[((n * 16 + c) * 10 + (oh * 2 + kh)) * 10 + (ow * 2 + kw)];
                        }
                    }
                    s4_out[((n * 16 + c) * 5 + oh) * 5 + ow] = sum / 4.0f;
                }
            }
        }
    }

    // --- STEP 5: FULLY CONNECTED LAYERS (GEMM + Tanh) ---
    auto perform_fc = [&](const std::vector<float>& in, int in_len, int out_len, 
                          size_t w_off, size_t b_off, float* out_ptr, bool apply_tanh) {
        for (int n = 0; n < opt.batch; ++n) {
            for (int j = 0; j < out_len; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < in_len; ++i) {
                    sum += in[n * in_len + i] * weights[w_off + j * in_len + i];
                }
                float val = sum + biases[b_off + j];
                out_ptr[n * out_len + j] = apply_tanh ? std::tanh(val) : val;
            }
        }
    };

    std::vector<float> fc1_out(shape.fc1_out_elems);
    perform_fc(s4_out, 400, shape.fc1_out, weight_offsets[2], bias_offsets[2], fc1_out.data(), true);

    std::vector<float> fc2_out(shape.fc2_out_elems);
    perform_fc(fc1_out, 120, shape.fc2_out, weight_offsets[3], bias_offsets[3], fc2_out.data(), true);

    // Final layer output usually does NOT have Tanh applied. May need to use use Softmax later. 
    perform_fc(fc2_out, 84, shape.fc3_out, weight_offsets[4], bias_offsets[4], output.data(), false);
}

// int main(int argc, char** argv) {
//     Options opt = parse_args(argc, argv);
//     LenetShape shape = make_lenet_shape(opt.batch);

//     std::vector<float> h_input(shape.input_elements);
//     std::vector<float> h_weights(shape.total_weight_elements);
//     std::vector<float> h_biases(shape.total_bias_elements);
//     std::vector<float> h_output(shape.output_elements, 0.0f);
//     std::vector<float> h_ref(shape.output_elements, 0.0f);

//     seed_tensor(h_input, 1.0f);
//     seed_tensor(h_weights, 0.05f);
//     seed_tensor(h_biases, 0.01f);

//     float* d_input = nullptr;
//     float* d_workspace = nullptr;
//     float* d_conv1_out = nullptr;
//     float* d_conv2_out = nullptr;
//     float* d_pool1_out = nullptr;
//     float* d_pool2_out = nullptr;
//     float* d_fc1_out = nullptr;
//     float* d_fc2_out = nullptr;
//     float* d_fc3_out = nullptr;
//     float* d_weights = nullptr;
//     float* d_biases = nullptr;
    
    
//     // Allocate handles and streams 
//     cudnnHandle_t cudnn;
//     check_cudnn(cudnnCreate(&cudnn), "cudnnCreate");
//     cublasHandle_t cublas;
//     check_cublas(cublasCreate(&cublas), "cublasCreate");

//     // Set up stream
//     cudaStream_t stream;
//     check_cuda(cudaStreamCreate(&stream), "create stream");
//     cudnnSetStream(cudnn, stream);
//     cublasSetStream(cublas, stream);

//     LenetDescriptors descs;
//     /* TODO(student): initialize tensor/filter/conv/pool descriptors using helpers in lenet_layers.cuh. */
//     create_lenet_descriptors(shape, descs);

//     /* TODO(student): cudaMalloc all required activation and weight buffers + copy host data. */

//     // 1. Allocate device buffers - inputs, weights, and biases
//     check_cuda(cudaMalloc(&d_input, shape.input_elements * sizeof(float)), "allocate d_input");
//     check_cuda(cudaMalloc(&d_weights, shape.total_weight_elements * sizeof(float)), "allocate d_weights");
//     check_cuda(cudaMalloc(&d_biases,  shape.total_bias_elements * sizeof(float)), "alloc d_biases");

//     // 2. Intermediate Activations (The "Working Memory")
//     // These are necessary because Layer 1 writes to d_conv1_out, then Layer 2 reads from it.
//     check_cuda(cudaMalloc(&d_conv1_out, shape.conv1_out_elems * sizeof(float)), "alloc d_conv1_out");
//     check_cuda(cudaMalloc(&d_pool1_out, shape.pool1_out_elems * sizeof(float)), "alloc d_pool1_out");
//     check_cuda(cudaMalloc(&d_conv2_out, shape.conv2_out_elems * sizeof(float)), "alloc d_conv2_out");
//     check_cuda(cudaMalloc(&d_pool2_out, shape.pool2_out_elems * sizeof(float)), "alloc d_pool2_out");

//     // 3. Fully Connected Outputs
//     check_cuda(cudaMalloc(&d_fc1_out,   shape.fc1_out_elems * sizeof(float)), "alloc d_fc1_out");
//     check_cuda(cudaMalloc(&d_fc2_out,   shape.fc2_out_elems * sizeof(float)), "alloc d_fc2_out");
//     check_cuda(cudaMalloc(&d_fc3_out,   shape.output_elements * sizeof(float)), "alloc d_fc3_out");
    
//     // 4. cuDNN workspace.
//     cudnnConvolutionFwdAlgo_t algo_desc = parse_algo(opt.algo);
//     size_t workspace_size_1 = query_conv_workspace(cudnn, shape, descs, algo_desc, false);
//     size_t workspace_size_2 = query_conv_workspace(cudnn, shape, descs, algo_desc, true);
//     size_t workspace_size = std::max(workspace_size_1, workspace_size_2);
//     check_cuda(cudaMalloc(&d_workspace, workspace_size), "alloc d_workspace");


//     // 5. Copy host data
//     check_cuda(cudaMemcpy(d_input, h_input.data(), shape.input_elements * sizeof(float), cudaMemcpyHostToDevice), "copy d_input");
//     check_cuda(cudaMemcpy(d_weights, h_weights.data(), shape.total_weight_elements * sizeof(float), cudaMemcpyHostToDevice), "copy d_weights");
//     check_cuda(cudaMemcpy(d_biases, h_biases.data(), shape.total_bias_elements * sizeof(float), cudaMemcpyHostToDevice), "copy d_biases");

   
   

//     cudaEvent_t start, stop;
//     check_cuda(cudaEventCreate(&start), "create start");
//     check_cuda(cudaEventCreate(&stop), "create stop");

//     float elapsed_ms = 0.0f;
//     if (opt.impl == "baseline") {
//         check_cuda(cudaEventRecord(start, stream), "record start baseline");
//         /* TODO(student):
//            1. run_lenet_conv for conv1/conv2 using opt.algo
//            2. launch_lenet_pool for each pooling stage
//            3. reshape tensor for FC input (either via dedicated kernel or by treating memory as-is)
//            4. run_fc_layer (cuBLAS GEMM + bias + activation) for the dense blocks
//         */

//         // ---  C1: Conv 1 (32x32 -> 28x28) ---
//         // false = first conv (uses input_desc and conv1_filter)
//         run_lenet_conv(cudnn, shape, descs, d_input, 
//                     d_weights + shape.weight_offsets[0], d_biases, shape.bias_offsets[0],
//                     d_conv1_out, d_workspace, workspace_size, opt.algo, false);

//         // --- S2: Pool 1 (28x28 -> 14x14) ---
//         // second_pool = false
//         run_lenet_pool(cudnn, descs, d_conv1_out, d_pool1_out, false);

//         // --- C3: Conv 2 (14x14 -> 10x10) ---
//         run_lenet_conv(cudnn, shape, descs, d_pool1_out, 
//                     d_weights + shape.weight_offsets[1], d_biases, shape.bias_offsets[1],
//                     d_conv2_out, d_workspace, workspace_size, opt.algo, true);

//         // --- S4: Pool 2 (10x10 -> 5x5) ---
//         // second_pool = true
//         run_lenet_pool(cudnn, descs, d_conv2_out, d_pool2_out, true);

//         // --- FC Layer 1 (400 -> 120) ---
//         run_fc_layer(cublas, shape, 0, d_pool2_out, 
//                     d_weights + shape.weight_offsets[2], d_biases + shape.bias_offsets[2], 
//                     d_fc1_out, stream);

//         // --- FC Layer 2 (120 -> 84) ---
//         run_fc_layer(cublas, shape, 1, d_fc1_out, 
//                     d_weights + shape.weight_offsets[3], d_biases + shape.bias_offsets[3], 
//                     d_fc2_out, stream);

//         // --- FC Layer 3 (84 -> 10) ---
//         run_fc_layer(cublas, shape, 2, d_fc2_out, 
//                     d_weights + shape.weight_offsets[4], d_biases + shape.bias_offsets[4], 
//                     d_fc3_out, stream);


//         check_cuda(cudaEventRecord(stop, stream), "record stop baseline");
//         check_cuda(cudaEventSynchronize(stop), "sync stop baseline");
//         check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed baseline");
//     } else if (opt.impl == "fused") {
//         check_cuda(cudaEventRecord(start, stream), "record start fused");
//         /* TODO(student): same as baseline but fuse activation/bias where possible. */
//         check_cuda(cudaEventRecord(stop, stream), "record stop fused");
//         check_cuda(cudaEventSynchronize(stop), "sync stop fused");
//         check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
//     } else {
//         throw std::invalid_argument("Unknown --impl=" + opt.impl);
//     }

//     /* TODO(student): copy logits from device to h_output. */
//     cudaMemcpyAsync(h_output.data(), d_fc3_out, shape.output_elements * sizeof(float), cudaMemcpyDeviceToHost, stream);
//     cudaStreamSynchronize(stream);

//     if (!opt.dump_path.empty()) {
//         std::ofstream ofs(opt.dump_path, std::ios::binary);
//         if (!ofs) {
//             throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
//         }
//         ofs.write(reinterpret_cast<const char*>(h_output.data()),
//                   static_cast<std::streamsize>(h_output.size() * sizeof(float)));
//         ofs.close();
//     }

//     float max_diff = 0.0f;
//     if (opt.verify) {
//         lenet_cpu_reference(opt,
//                             shape,
//                             h_weights,
//                             shape.weight_offsets,
//                             h_biases,
//                             shape.bias_offsets,
//                             h_input,
//                             h_ref);
//         /* TODO(student): compute and print max abs diff between h_output and h_ref. */
//          for (size_t i = 0; i < h_ref.size(); ++i) {
//             max_diff = std::max(max_diff, std::abs(h_output[i] - h_ref[i]));
//         }
//         std::cout << "Max absolute difference: " << max_diff << std::endl;
//     }

//     if (elapsed_ms > 0.0f) {
//         std::cout << std::fixed << std::setprecision(2)
//                   << "Impl=" << opt.impl
//                   << " Batch=" << opt.batch
//                   << " Algo=" << opt.algo
//                   << " Time(ms)=" << elapsed_ms
//                   << " GFLOP/s=" << lenet_gflops(shape, elapsed_ms) << std::endl;
//     } else {
//         std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
//     }

//     /* TODO(student): destroy descriptors, handles, free device buffers, destroy events. */
    
//     // 1. Library Cleanup
//     check_cudnn(cudnnDestroy(cudnn), "destroy cudnn");
//     check_cublas(cublasDestroy(cublas), "destroy cublas");

//     // 2. Descriptor Cleanup 
//     destroy_lenet_descriptors(descs);

//     // 3. Buffer Cleanup
//     cudaFree(d_input); cudaFree(d_weights); cudaFree(d_biases); cudaFree(d_workspace);
//     cudaFree(d_conv1_out); cudaFree(d_conv2_out); cudaFree(d_pool1_out); cudaFree(d_pool2_out);
//     cudaFree(d_fc1_out); cudaFree(d_fc2_out); cudaFree(d_fc3_out);

//     // 4. Synchronization/Event Cleanup
//     cudaEventDestroy(start); cudaEventDestroy(stop);
//     cudaStreamDestroy(stream);

//     return 0;
// }

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    LenetShape shape = make_lenet_shape(opt.batch);

    std::vector<float> h_input(shape.input_elements);
    std::vector<float> h_weights(shape.total_weight_elements);
    std::vector<float> h_biases(shape.total_bias_elements);
    std::vector<float> h_output(shape.output_elements, 0.0f);
    std::vector<float> h_ref(shape.output_elements, 0.0f);

    seed_tensor(h_input, 1.0f);
    seed_tensor(h_weights, 0.05f);
    seed_tensor(h_biases, 0.01f);

    float* d_input = nullptr;
    float* d_workspace = nullptr;
    float* d_conv1_out = nullptr;
    float* d_conv2_out = nullptr;
    float* d_pool1_out = nullptr;
    float* d_pool2_out = nullptr;
    float* d_fc1_out = nullptr;
    float* d_fc2_out = nullptr;
    float* d_fc3_out = nullptr;
    float* d_weights = nullptr;
    float* d_biases = nullptr;
    
    
    // Allocate handles and streams 
    cudnnHandle_t cudnn;
    check_cudnn(cudnnCreate(&cudnn), "cudnnCreate");
    cublasHandle_t cublas;
    check_cublas(cublasCreate(&cublas), "cublasCreate");

    // Set up stream
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create stream");
    cudnnSetStream(cudnn, stream);
    cublasSetStream(cublas, stream);

    LenetDescriptors descs;
    /* TODO(student): initialize tensor/filter/conv/pool descriptors using helpers in lenet_layers.cuh. */
    create_lenet_descriptors(shape, descs);

    /* TODO(student): cudaMalloc all required activation and weight buffers + copy host data. */

    // 1. Allocate device buffers - inputs, weights, and biases
    check_cuda(cudaMalloc(&d_input, shape.input_elements * sizeof(float)), "allocate d_input");
    check_cuda(cudaMalloc(&d_weights, shape.total_weight_elements * sizeof(float)), "allocate d_weights");
    check_cuda(cudaMalloc(&d_biases,  shape.total_bias_elements * sizeof(float)), "alloc d_biases");

    // 2. Intermediate Activations (The "Working Memory")
    // These are necessary because Layer 1 writes to d_conv1_out, then Layer 2 reads from it.
    check_cuda(cudaMalloc(&d_conv1_out, shape.conv1_out_elems * sizeof(float)), "alloc d_conv1_out");
    check_cuda(cudaMalloc(&d_pool1_out, shape.pool1_out_elems * sizeof(float)), "alloc d_pool1_out");
    check_cuda(cudaMalloc(&d_conv2_out, shape.conv2_out_elems * sizeof(float)), "alloc d_conv2_out");
    check_cuda(cudaMalloc(&d_pool2_out, shape.pool2_out_elems * sizeof(float)), "alloc d_pool2_out");

    // 3. Fully Connected Outputs
    check_cuda(cudaMalloc(&d_fc1_out,   shape.fc1_out_elems * sizeof(float)), "alloc d_fc1_out");
    check_cuda(cudaMalloc(&d_fc2_out,   shape.fc2_out_elems * sizeof(float)), "alloc d_fc2_out");
    check_cuda(cudaMalloc(&d_fc3_out,   shape.output_elements * sizeof(float)), "alloc d_fc3_out");
    
    // 4. cuDNN workspace.
    cudnnConvolutionFwdAlgo_t algo_desc = parse_algo(opt.algo);
    size_t workspace_size_1 = query_conv_workspace(cudnn, shape, descs, algo_desc, false);
    size_t workspace_size_2 = query_conv_workspace(cudnn, shape, descs, algo_desc, true);
    size_t workspace_size = std::max(workspace_size_1, workspace_size_2);
    check_cuda(cudaMalloc(&d_workspace, workspace_size), "alloc d_workspace");


    // 5. Copy host data
    check_cuda(cudaMemcpy(d_input, h_input.data(), shape.input_elements * sizeof(float), cudaMemcpyHostToDevice), "copy d_input");
    check_cuda(cudaMemcpy(d_weights, h_weights.data(), shape.total_weight_elements * sizeof(float), cudaMemcpyHostToDevice), "copy d_weights");
    check_cuda(cudaMemcpy(d_biases, h_biases.data(), shape.total_bias_elements * sizeof(float), cudaMemcpyHostToDevice), "copy d_biases");

    // --- WARMUP RUN (Untimed) ---
    // This handles library initialization and kernel loading outside the timer.
    for (int i = 0; i < 5; ++i) {
        if (opt.impl == "baseline") {
            run_lenet_conv(cudnn, shape, descs, d_input, d_weights + shape.weight_offsets[0], d_biases, shape.bias_offsets[0], d_conv1_out, d_workspace, workspace_size, opt.algo, false);
            run_lenet_pool(cudnn, descs, d_conv1_out, d_pool1_out, false);
            run_lenet_conv(cudnn, shape, descs, d_pool1_out, d_weights + shape.weight_offsets[1], d_biases, shape.bias_offsets[1], d_conv2_out, d_workspace, workspace_size, opt.algo, true);
            run_lenet_pool(cudnn, descs, d_conv2_out, d_pool2_out, true);
            run_fc_layer(cublas, shape, 0, d_pool2_out, d_weights + shape.weight_offsets[2], d_biases + shape.bias_offsets[2], d_fc1_out, stream);
            run_fc_layer(cublas, shape, 1, d_fc1_out, d_weights + shape.weight_offsets[3], d_biases + shape.bias_offsets[3], d_fc2_out, stream);
            run_fc_layer(cublas, shape, 2, d_fc2_out, d_weights + shape.weight_offsets[4], d_biases + shape.bias_offsets[4], d_fc3_out, stream);
        }
    }
    check_cuda(cudaDeviceSynchronize(), "Warmup sync");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    const int iterations = 100;
    float total_elapsed_ms = 0.0f;
    float elapsed_ms = 0.0f;

    if (opt.impl == "baseline") {
        check_cuda(cudaEventRecord(start, stream), "record start baseline");
        /* TODO(student):
           1. run_lenet_conv for conv1/conv2 using opt.algo
           2. launch_lenet_pool for each pooling stage
           3. reshape tensor for FC input (either via dedicated kernel or by treating memory as-is)
           4. run_fc_layer (cuBLAS GEMM + bias + activation) for the dense blocks
        */

        for (int i = 0; i < iterations; ++i) {
            // ---  C1: Conv 1 (32x32 -> 28x28) ---
            // false = first conv (uses input_desc and conv1_filter)
            run_lenet_conv(cudnn, shape, descs, d_input, 
                        d_weights + shape.weight_offsets[0], d_biases, shape.bias_offsets[0],
                        d_conv1_out, d_workspace, workspace_size, opt.algo, false);

            // --- S2: Pool 1 (28x28 -> 14x14) ---
            // second_pool = false
            run_lenet_pool(cudnn, descs, d_conv1_out, d_pool1_out, false);

            // --- C3: Conv 2 (14x14 -> 10x10) ---
            run_lenet_conv(cudnn, shape, descs, d_pool1_out, 
                        d_weights + shape.weight_offsets[1], d_biases, shape.bias_offsets[1],
                        d_conv2_out, d_workspace, workspace_size, opt.algo, true);

            // --- S4: Pool 2 (10x10 -> 5x5) ---
            // second_pool = true
            run_lenet_pool(cudnn, descs, d_conv2_out, d_pool2_out, true);

            // --- FC Layer 1 (400 -> 120) ---
            run_fc_layer(cublas, shape, 0, d_pool2_out, 
                        d_weights + shape.weight_offsets[2], d_biases + shape.bias_offsets[2], 
                        d_fc1_out, stream);

            // --- FC Layer 2 (120 -> 84) ---
            run_fc_layer(cublas, shape, 1, d_fc1_out, 
                        d_weights + shape.weight_offsets[3], d_biases + shape.bias_offsets[3], 
                        d_fc2_out, stream);

            // --- FC Layer 3 (84 -> 10) ---
            run_fc_layer(cublas, shape, 2, d_fc2_out, 
                        d_weights + shape.weight_offsets[4], d_biases + shape.bias_offsets[4], 
                        d_fc3_out, stream);
        }

        check_cuda(cudaEventRecord(stop, stream), "record stop baseline");
        check_cuda(cudaEventSynchronize(stop), "sync stop baseline");
        check_cuda(cudaEventElapsedTime(&total_elapsed_ms, start, stop), "elapsed baseline");
        elapsed_ms = total_elapsed_ms / iterations;
    } else if (opt.impl == "fused") {
        check_cuda(cudaEventRecord(start, stream), "record start fused");
        /* TODO(student): same as baseline but fuse activation/bias where possible. */
        for (int i = 0; i < iterations; ++i) {
            // Fused implementation logic here
        }
        check_cuda(cudaEventRecord(stop, stream), "record stop fused");
        check_cuda(cudaEventSynchronize(stop), "sync stop fused");
        check_cuda(cudaEventElapsedTime(&total_elapsed_ms, start, stop), "elapsed fused");
        elapsed_ms = total_elapsed_ms / iterations;
    } else {
        throw std::invalid_argument("Unknown --impl=" + opt.impl);
    }

    /* TODO(student): copy logits from device to h_output. */
    cudaMemcpyAsync(h_output.data(), d_fc3_out, shape.output_elements * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (!opt.dump_path.empty()) {
        std::ofstream ofs(opt.dump_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
        }
        ofs.write(reinterpret_cast<const char*>(h_output.data()),
                  static_cast<std::streamsize>(h_output.size() * sizeof(float)));
        ofs.close();
    }

    float max_diff = 0.0f;
    if (opt.verify) {
        lenet_cpu_reference(opt,
                            shape,
                            h_weights,
                            shape.weight_offsets,
                            h_biases,
                            shape.bias_offsets,
                            h_input,
                            h_ref);
        /* TODO(student): compute and print max abs diff between h_output and h_ref. */
         for (size_t i = 0; i < h_ref.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(h_output[i] - h_ref[i]));
        }
        std::cout << "Max absolute difference: " << max_diff << std::endl;
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2)
                  << "Impl=" << opt.impl
                  << " Batch=" << opt.batch
                  << " Algo=" << opt.algo
                  << " Time(ms)=" << elapsed_ms
                  << " GFLOP/s=" << lenet_gflops(shape, elapsed_ms) << std::endl;
    } else {
        std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
    }

    /* TODO(student): destroy descriptors, handles, free device buffers, destroy events. */
    
    // 1. Library Cleanup
    check_cudnn(cudnnDestroy(cudnn), "destroy cudnn");
    check_cublas(cublasDestroy(cublas), "destroy cublas");

    // 2. Descriptor Cleanup 
    destroy_lenet_descriptors(descs);

    // 3. Buffer Cleanup
    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_biases); cudaFree(d_workspace);
    cudaFree(d_conv1_out); cudaFree(d_conv2_out); cudaFree(d_pool1_out); cudaFree(d_pool2_out);
    cudaFree(d_fc1_out); cudaFree(d_fc2_out); cudaFree(d_fc3_out);

    // 4. Synchronization/Event Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return 0;
}
