// #pragma once

// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <cmath>

// #include <string>
// #include <vector>

// struct LayerShape {
//     int batch;
//     int in_dim;
//     int out_dim;
// };

// inline double layer_flops(const LayerShape& shape) {
//     return 2.0 * static_cast<double>(shape.batch) * shape.in_dim * shape.out_dim;
// }

// inline double mlp_gflops(const std::vector<int>& layers, int batch, double millis) {
//     double total_flops = 0.0;
//     for (size_t i = 0; i + 1 < layers.size(); ++i) {
//         LayerShape shape{batch, layers[i], layers[i + 1]};
//         total_flops += layer_flops(shape);
//     }
//     return total_flops / (millis * 1e6);
// }

// __global__ void bias_add_kernel(const float* __restrict__ bias,
//                                 float* __restrict__ activations,
//                                 LayerShape shape) {
//     /* TODO(student): each thread should add the bias for its neuron across the batch. */

//     int row = blockIdx.y * blockDim.y + threadIdx.y; // Batch index
//     int col = blockIdx.x * blockDim.x + threadIdx.x; // Neuron index

//     if (row < shape.batch && col < shape.out_dim) {
//         size_t idx = (size_t)row * shape.out_dim + col;
//         activations[idx] += bias[col];
//     }
// }

// __global__ void relu_kernel(float* __restrict__ activations, size_t elements) {
//     /* TODO(student): ReLU activation (set negative values to zero). */

//     // Calculate the global col index of the current thread.
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;

//     // Bounday check
//     if (index < elements) {
//         // Apply ReLU activation: output is the maximum of 0.0f and the input value.
//         activations[index] = max(0.0f, activations[index]);
//     }
// }

// __global__ void gelu_kernel(float* __restrict__ activations, size_t elements) {
//     /* TODO(student): Approximate GELU, e.g., 0.5*x*(1+tanh(...)). */

//      // Calculate the global col index of the current thread.
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;

//     // Bounday check
//     if (index < elements) {
//         float x = activations[index];
//         float sqrt_2_over_pi = 0.7978845608f;

//         // Apply GELU activation: 
//         activations[index] = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * ( x * x * x))));
//     }

// }

// inline void launch_bias_add(const float* bias, float* activations, const LayerShape& shape, cudaStream_t stream) {
//     const int threads = 256;
//     const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
//     const int blocks = static_cast<int>((elements + threads - 1) / threads);
//     bias_add_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape);
//     (void)elements;  // silence unused warnings until kernel implemented
// }

// inline void launch_activation(const std::string& activation,
//                               float* activations,
//                               const LayerShape& shape,
//                               cudaStream_t stream) {
//     const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
//     const int threads = 256;
//     const int blocks = static_cast<int>((elements + threads - 1) / threads);
//     if (activation == "relu") {
//         relu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
//     } else if (activation == "gelu") {
//         gelu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
//     } else {
//         // TODO(student): add more activations as desired
//     }
// }

// __global__ void fused_bias_activation_kernel(const float* __restrict__ bias,
//                                              float* __restrict__ activations,
//                                              LayerShape shape,
//                                              int activation_type) {
//     /* TODO(student): fuse bias add + activation.
//        activation_type: 0=ReLU, 1=GELU, extend as needed. */

//     // Init (1D)
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;

//     // Boundary guard
//     if (index < (size_t) shape.batch * shape.out_dim) {
//         // 2D -> 1D index mapping
//         size_t col = index % shape.out_dim; // Which neuron


//         // Apply bias - all threads in the same column use the same bias[col]
//         float val = activations[index] + bias[col];

//         // Chosen activation function
//         if (activation_type == 0){
//             // ReLU
//             val = max(0.0f, val);
//         } else if (activation_type == 1) {
//             // GELU
//             float sqrt_2_over_pi = 0.7978845608f;
//             val =  0.5f * val * (1.0f + tanhf(sqrt_2_over_pi * (val + 0.044715f * ( val * val * val))));
//         }

//         // Final write-back
//         activations[index] = val;
//     }
// }

// inline void launch_fused_bias_activation(const float* bias,
//                                          const std::string& activation,
//                                          float* activations,
//                                          const LayerShape& shape,
//                                          cudaStream_t stream) {
//     int activation_type = (activation == "gelu") ? 1 : 0;
//     const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
//     const int threads = 256;
//     const int blocks = static_cast<int>((elements + threads - 1) / threads);
//     fused_bias_activation_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape, activation_type);
//     (void)elements;
// }

// inline void run_gemm_layer(const float* input,
//                            const float* weight,
//                            float* output,
//                            const LayerShape& shape,
//                            cublasHandle_t handle) {
//     /* TODO(student): call cublasSgemm (or StridedBatched) with the correct transpose options.
//        Remember cuBLAS assumes column-major by default; consider using CUBLAS_OP_T to match row-major data. */
       
//     const float alpha = 1.0f;
//     const float beta = 0.0f;

//     cublasSgemm(handle, 
//             CUBLAS_OP_T,     // Transpose Weight (Row-major [O, I] -> Col-major [I, O])
//             CUBLAS_OP_N,     // No Transpose Input (Row-major [B, I] -> Col-major [I, B])
//             shape.out_dim,   // M: Rows of Op(A)
//             shape.batch,     // N: Cols of Op(B)
//             shape.in_dim,    // K: Cols of Op(A)
//             &alpha, 
//             weight,          
//             shape.in_dim,    // lda: MUST be >= in_dim when transposing
//             input,           
//             shape.in_dim,    // ldb: Row-major [B, I] is Col-major [I, B], so leading dim is in_dim
//             &beta, 
//             output,          
//             shape.out_dim);  // ldc: Leading dim is out_dim

// }


// v2


#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>

#include <string>
#include <vector>

struct LayerShape {
    int batch;
    int in_dim;
    int out_dim;
};

inline double layer_flops(const LayerShape& shape) {
    return 2.0 * static_cast<double>(shape.batch) * shape.in_dim * shape.out_dim;
}

inline double mlp_gflops(const std::vector<int>& layers, int batch, double millis) {
    double total_flops = 0.0;
    for (size_t i = 0; i + 1 < layers.size(); ++i) {
        LayerShape shape{batch, layers[i], layers[i + 1]};
        total_flops += layer_flops(shape);
    }
    return total_flops / (millis * 1e6);
}

__global__ void bias_add_kernel(const float* __restrict__ bias,
                                float* __restrict__ activations,
                                LayerShape shape) {
    /* TODO(student): each thread should add the bias for its neuron across the batch. */
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Neuron index

    if (row < shape.batch && col < shape.out_dim) {
        size_t idx = (size_t)row * shape.out_dim + col;
        activations[idx] += bias[col];
    }
}

__global__ void relu_kernel(float* __restrict__ activations, size_t elements) {
    /* TODO(student): ReLU activation (set negative values to zero). */
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        activations[index] = max(0.0f, activations[index]);
    }
}

__global__ void gelu_kernel(float* __restrict__ activations, size_t elements) {
    /* TODO(student): Approximate GELU, e.g., 0.5*x*(1+tanh(...)). */
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        float x = activations[index];
        float sqrt_2_over_pi = 0.7978845608f;
        activations[index] = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * ( x * x * x))));
    }
}

inline void launch_bias_add(const float* bias, float* activations, const LayerShape& shape, cudaStream_t stream) {
    const int threads = 256;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    bias_add_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape);
    (void)elements; 
}

inline void launch_activation(const std::string& activation,
                              float* activations,
                              const LayerShape& shape,
                              cudaStream_t stream) {
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    if (activation == "relu") {
        relu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
    } else if (activation == "gelu") {
        gelu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
    }
}

__global__ void fused_bias_activation_kernel(const float* __restrict__ bias,
                                             float* __restrict__ activations,
                                             LayerShape shape,
                                             int activation_type) {
    /* TODO(student): fuse bias add + activation.
       activation_type: 0=ReLU, 1=GELU, extend as needed. */
    
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = (size_t)shape.batch * shape.out_dim;

    if (index < total_elements) {
        // Optimization: Replace modulo (%) with fixed-point math or simpler logic 
        // to find the bias index (column index) while keeping the 1D launch.
        size_t col = index % shape.out_dim; 

        float val = activations[index] + bias[col];

        if (activation_type == 0) {
            val = max(0.0f, val);
        } else if (activation_type == 1) {
            const float sqrt_2_over_pi = 0.7978845608f;
            val = 0.5f * val * (1.0f + tanhf(sqrt_2_over_pi * (val + 0.044715f * (val * val * val))));
        }
        activations[index] = val;
    }
}

inline void launch_fused_bias_activation(const float* bias,
                                         const std::string& activation,
                                         float* activations,
                                         const LayerShape& shape,
                                         cudaStream_t stream) {
    int activation_type = (activation == "gelu") ? 1 : 0;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    fused_bias_activation_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape, activation_type);
    (void)elements;
}

inline void run_gemm_layer(const float* input,
                           const float* weight,
                           float* output,
                           const LayerShape& shape,
                           cublasHandle_t handle) {
    /* TODO(student): call cublasSgemm (or StridedBatched) with the correct transpose options. */
       
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // A: Weight [Out, In] (Row-major) -> Treated as [In, Out] (Col-major)
    // B: Input  [Batch, In] (Row-major) -> Treated as [In, Batch] (Col-major)
    // C: Output [Batch, Out] (Row-major) -> Treated as [Out, Batch] (Col-major)
    
    // To get Output = Input * Weight^T:
    // We compute C^T = (Input * Weight^T)^T = Weight * Input^T
    
   cublasSgemm(handle, 
                CUBLAS_OP_T,     // Transpose A (Weight): [Out, In] Row -> [In, Out] Col
                CUBLAS_OP_N,     // No Transpose B (Input): [Batch, In] Row is [In, Batch] Col
                shape.out_dim,   // M: Rows of Op(A) and C
                shape.batch,     // N: Columns of Op(B) and C
                shape.in_dim,    // K: Columns of Op(A) / Rows of Op(B)
                &alpha, 
                weight, 
                shape.in_dim,    // lda: row-width of weight
                input, 
                shape.in_dim,    // ldb: row-width of input
                &beta, 
                output, 
                shape.out_dim);  // ldc: row-width of output
                
}