#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "conv_kernel.cuh"

struct Options {
    int height = 256;
    int width = 256;
    int channels = 32;
    int filters = 64;
    int kernel = 3;
    int stride = 1;
    int padding = 1;
    std::string impl = "baseline";  // baseline==naive GPU, tiled==shared memory
    bool verify = true;
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--height") == 0 || strcmp(argv[i], "-H") == 0) && i + 1 < argc) {
            opt.height = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--width") == 0 || strcmp(argv[i], "-W") == 0) && i + 1 < argc) {
            opt.width = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
            opt.channels = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--filters") == 0 && i + 1 < argc) {
            opt.filters = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--ksize") == 0 || strcmp(argv[i], "-K") == 0) && i + 1 < argc) {
            opt.kernel = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--stride") == 0 && i + 1 < argc) {
            opt.stride = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--padding") == 0 && i + 1 < argc) {
            opt.padding = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dconv2d --height H --width W --channels Cin --filters Cout \\\n  --ksize K --stride S --padding P --impl baseline|naive|tiled [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void seed_tensor(std::vector<float>& data, float scale) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = scale * std::sin(0.13f * static_cast<float>(i));
    }
}

void conv2d_cpu_reference(const Conv2dShape& shape,
                          const std::vector<float>& input,
                          const std::vector<float>& weight,
                          std::vector<float>& output) {
    std::fill(output.begin(), output.end(), 0.0f);
    /* TODO(student): implement the nested loops over Cout, Hout, Wout, Cin, and KxK.
       Use shape helpers (see conv_kernel.cuh) to translate indices and honor padding. */

    // 1. Loop over every output channel (filter)
    for (int oc = 0; oc < shape.filters; oc++) {
         
        // 2. Loop over every row of the output image
         for (int oh = 0; oh < shape.out_height; oh++) {

            // 3. Loop over every column of the output image
            for (int ow = 0; ow < shape.out_width; ow++) {

                // Accumulator
                float acc = 0.0f;

                // 4. Loop over every input channel
                for (int ic = 0; ic < shape.channels; ic++) {
                    
                    // 5 AND 6. Loop over the math kernl; (Filter Width/Height). Calculate  corresponding input coordinates (ih, iw) and boundary check for padding
                    for (int kh = 0; kh < shape.kernel; kh++) {
                        for (int kw = 0; kw < shape.kernel; kw++) {
                            // Calculate the corresponding coordinate in the INPUT image
                            int ih = oh * shape.stride - shape.padding + kh;
                            int iw = ow * shape.stride - shape.padding + kw;
                            
                            // Skip taps that fall outside the padded image.
                            if (iw >= 0 && iw < shape.width && ih >= 0 && ih < shape.height) {
                                int in_idx = input_index(shape, ic, ih, iw);
                                int w_idx = weight_index(shape, oc, ic, kh, kw);
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }
                // 7. Write the final sum to the output tensor
                output[output_index(shape, oc, oh, ow)] = acc;
            }
        }
    }
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    Conv2dShape shape = make_shape(opt.height, opt.width, opt.channels, opt.filters, opt.kernel, opt.stride, opt.padding);
    const size_t input_elems = static_cast<size_t>(shape.channels) * shape.height * shape.width;
    const size_t weight_elems = static_cast<size_t>(shape.filters) * shape.channels * shape.kernel * shape.kernel;
    const size_t output_elems = static_cast<size_t>(shape.filters) * shape.out_height * shape.out_width;

    std::vector<float> h_input(input_elems);
    std::vector<float> h_weight(weight_elems);
    std::vector<float> h_output(output_elems, 0.0f);
    std::vector<float> h_ref(output_elems, 0.0f);

    seed_tensor(h_input, 1.0f);
    seed_tensor(h_weight, 0.5f);
    /* TODO(student): consider adding bias tensors or different initializations for experiments. */

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;
    /* TODO(student): cudaMalloc device buffers and copy host data (cudaMemcpy H2D). */
    // (void)d_input;
    // (void)d_weight;
    // (void)d_output;

    check_cuda(cudaMalloc(&d_input, input_elems * sizeof(float)), "allocate d_input");
    check_cuda(cudaMalloc(&d_weight, weight_elems * sizeof(float)), "allocate d_weight");
    check_cuda(cudaMalloc(&d_output, output_elems * sizeof(float)), "allocate d_output");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create stream");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventRecord(start, stream), "record start");

    if (opt.impl == "baseline" || opt.impl == "naive") {
        /* TODO(student): record events around launch_naive_conv2d and compute elapsed_ms. */
        launch_naive_conv2d(d_input, d_weight, d_output, shape, stream);
    } else if (opt.impl == "tiled") {
        /* TODO(student): time the shared-memory kernel via launch_tiled_conv2d. */
        launch_tiled_conv2d(d_input, d_weight, d_output, shape, stream);
    } else {
        throw std::invalid_argument("Unknown implementation: " + opt.impl);
    }

    check_cuda(cudaEventRecord(stop, stream), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed total");

    /* TODO(student): copy device output back to h_output (cudaMemcpy D2H). */
    cudaMemcpyAsync(h_output.data(), d_output, output_elems * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float max_diff = 0.0f;
    if (opt.verify) {
        conv2d_cpu_reference(shape, h_input, h_weight, h_ref);
        /* TODO(student): compute max absolute error between h_output and h_ref. */
        for (size_t i = 0; i < h_ref.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(h_output[i] - h_ref[i]));
        }
        std::cout << "Max absolute difference: " << max_diff << std::endl;
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Impl=" << opt.impl << " H=" << shape.height << " W=" << shape.width
                  << " Cin=" << shape.channels << " Cout=" << shape.filters
                  << " K=" << shape.kernel << " Time(ms)=" << elapsed_ms
                  << " GFLOP/s=" << conv_gflops(shape, elapsed_ms) << std::endl;
    } else {
        std::cout << "Impl=" << opt.impl << " executed (timing TODO not yet implemented)" << std::endl;
    }

    /* TODO(student): free device memory and destroy CUDA events/streams. */
    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return 0;
}
