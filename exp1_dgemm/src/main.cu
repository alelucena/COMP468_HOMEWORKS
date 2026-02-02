#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm_kernel.cuh"

struct Options {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    std::string impl = "baseline";
    bool verify = true;
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--m") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            opt.m = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--n") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            opt.n = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--k") == 0 || strcmp(argv[i], "-k") == 0) && i + 1 < argc) {
            opt.k = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dgemm [--m int] [--n int] [--k int] [--impl baseline|naive|tiled|cublas] [--no-verify]\n";
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

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

double gflops(int m, int n, int k, double millis) {
    double flops = 2.0 * m * n * k;
    return flops / (millis * 1e6);
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    const int m = opt.m, n = opt.n, k = opt.k;
    const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(float);

    // h_a(m * k): Creates a vector for Matrix A with enough space for m * k floating-point numbers.
    // h_b(k * n): Creates a vector for Matrix B with space for k * n floats.

    // h_c(m * n, 0.0f): Creates a vector for the GPU's output (Matrix C). The second argument,
    // 0.0f, ensures every single element is initialized to zero.

    // h_ref(m * n, 0.0f): Creates a vector for the "Reference" result (calculated by the CPU). 
    // Used to check if the GPU math is actually correct.

    std::vector<float> h_a(m * k), h_b(k * n), h_c(m * n, 0.0f), h_ref(m * n, 0.0f);

    /* TODO(student): initialize h_a, h_b with reproducible random data (e.g., std::sin / std::cos). */
    for (size_t i = 0; i < m * k; i++) {
        h_a[i] = std::sin(static_cast<float>(i));
    }

    for (size_t i = 0; i < k * n; i++) {
        h_b[i] = std::cos(static_cast<float>(i));
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    /* TODO(student): allocate device buffers and copy host data over. */

    // Allocate device buffers
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    // Copy data to GPU from host
    cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data() , bytes_b, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline" || opt.impl == "naive" || opt.impl == "tiled") {
        /* TODO(student): choose the right launch helper based on opt.impl and record elapsed_ms. */

        // CUDA events for timing
        cudaEventRecord(start);

        // Select the appropriate launcher - default stream (0)
        if (opt.impl == "baseline" ||  opt.impl == "naive") {
            launch_naive_gemm(d_a, d_b, d_c, opt.m, opt.n, opt.k, 0);
        } else if (opt.impl == "tiled") {
            launch_tiled_gemm(d_a, d_b, d_c, opt.m, opt.n, opt.k, 0);
        }

        // Record Stop and elapsed time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);

    } else if (opt.impl == "cublas") {
        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "cublasCreate");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        check_cuda(cudaEventRecord(start), "record start");
        check_cublas(
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        d_b,
                        n,
                        d_a,
                        k,
                        &beta,
                        d_c,
                        n),
            "cublasSgemm");
        check_cuda(cudaEventRecord(stop), "record stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
        check_cublas(cublasDestroy(handle), "cublasDestroy");
    } else {
        throw std::invalid_argument("Unknown implementation: " + opt.impl);
    }

    /* TODO(student): copy d_c back into h_c. */
    cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

    if (opt.verify) {
        /* TODO(student): run cuBLAS reference into h_ref (or reuse above) and compute max error. */

        cublasHandle_t handle;
        cublasCreate(&handle);

        // 2. Call SGEMM (Single-precision General Matrix Multiply) - Column Major
        // C =  alpha*A*B + beta*C
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    opt.n, opt.m, opt.k,
                    &alpha,
                    d_b, opt.n,
                    d_a, opt.k,
                    &beta,
                    d_c, opt.n);
        
        // Copy cuBLAS result to h_ref
        cudaMemcpy(h_ref.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);
        cublasDestroy(handle);

        // Compute max error
        float max_error = 0.0f;
        for (int i = 0; i < m * n; i++) {
            float abs_difference = std::abs(h_c[i] - h_ref[i]);
            if (abs_difference > max_error) {
                max_error = abs_difference;
            }
        }

        printf("Max Error: %e\n", max_error);


    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Impl=" << opt.impl << " M=" << m << " N=" << n << " K=" << k
                  << " Time(ms)=" << elapsed_ms << " GFLOP/s=" << gflops(m, n, k, elapsed_ms)
                  << std::endl;
    }

    /* TODO(student): free device memory and destroy CUDA events. */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

