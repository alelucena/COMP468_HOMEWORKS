#pragma once

#include <cuda_runtime.h>

struct Conv2dShape {
    int height;
    int width;
    int channels;
    int filters;
    int kernel;
    int stride;
    int padding;
    int out_height;
    int out_width;
};

inline Conv2dShape make_shape(int height,
                              int width,
                              int channels,
                              int filters,
                              int kernel,
                              int stride,
                              int padding) {
    Conv2dShape shape{height,
                      width,
                      channels,
                      filters,
                      kernel,
                      stride,
                      padding,
                      0,
                      0};
    shape.out_height = (height + 2 * padding - kernel) / stride + 1;
    shape.out_width = (width + 2 * padding - kernel) / stride + 1;
    return shape;
}

inline __host__ __device__ int input_index(const Conv2dShape& shape, int c, int h, int w) {
    return (c * shape.height + h) * shape.width + w;
}

inline __host__ __device__ int weight_index(const Conv2dShape& shape, int oc, int ic, int kh, int kw) {
    return ((oc * shape.channels + ic) * shape.kernel + kh) * shape.kernel + kw;
}

inline __host__ __device__ int output_index(const Conv2dShape& shape, int oc, int oh, int ow) {
    return (oc * shape.out_height + oh) * shape.out_width + ow;
}

constexpr int BLOCK_SIZE = 16;

inline dim3 make_conv_grid(const Conv2dShape& shape) {
    return dim3((shape.out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (shape.out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                shape.filters);
}

__global__ void conv2d_naive_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output,
                                    Conv2dShape shape) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z; // output channel
    if (ow >= shape.out_width || oh >= shape.out_height || oc >= shape.filters) {
        return;
    }

    float acc = 0.0f;
    /* TODO(student): loop over channels/ksize and accumulate into acc. Remember padding offsets:
       ih = oh * stride - padding + kh;
       iw = ow * stride - padding + kw;
       Skip taps that fall outside the padded image. */

    for (int ic = 0; ic < shape.channels; ic++) {
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
    output[output_index(shape, oc, oh, ow)] = acc;
}

__global__ void conv2d_tiled_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output,
                                    Conv2dShape shape) {
    extern __shared__ float tile[];
    const int K = shape.kernel;
    const int SHARED_DIM = BLOCK_SIZE + K - 1; 
    
    float* tile_input = tile;                              
    float* tile_weight = tile + (SHARED_DIM * SHARED_DIM);   

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int oc = blockIdx.z; 

    // Define output coordinates once
    const int oh = blockIdx.y * BLOCK_SIZE + ty;
    const int ow = blockIdx.x * BLOCK_SIZE + tx;

    // The top-left corner of the INPUT area this block needs (including padding offset)
    const int row_start = blockIdx.y * BLOCK_SIZE * shape.stride - shape.padding;
    const int col_start = blockIdx.x * BLOCK_SIZE * shape.stride - shape.padding;

    float acc = 0.0f; 
    
    // Loop over each channel
    for (int ic = 0; ic < shape.channels; ic++) {

        // 1. Fill the tile_input (Collaborative Load)
        int total_tile_elements = SHARED_DIM * SHARED_DIM;
        int block_threads = BLOCK_SIZE * BLOCK_SIZE;
        int thread_id = ty * BLOCK_SIZE + tx;

        for (int tid = thread_id; tid < total_tile_elements; tid += block_threads) {
            int i = tid / SHARED_DIM;
            int j = tid % SHARED_DIM;
            int curr_row = row_start + i;
            int curr_col = col_start + j;

            if (curr_row >= 0 && curr_row < shape.height && curr_col >= 0 && curr_col < shape.width) {
                tile_input[i * SHARED_DIM + j] = input[input_index(shape, ic, curr_row, curr_col)];
            } else {
                // Padding if out-of-bounds
                tile_input[i * SHARED_DIM + j] = 0.0f; 
            }
        }

        // 2. Load weights for this specific (output_channel, input_channel)
        if (ty < K && tx < K) {
            tile_weight[ty * K + tx] = weight[weight_index(shape, oc, ic, ty, tx)];
        }

        // SYNC: Ensure everyone has finished loading before starting math
        __syncthreads();

        // 3. Compute
        if (oh < shape.out_height && ow < shape.out_width) {
            #pragma unroll // optimize the inner loops into line code,
            for (int kh = 0; kh < K; kh++) {
                #pragma unroll
                for (int kw = 0; kw < K; kw++) {
                    // Anchor at (ty*S, tx*S) and offset by kernel index (kh, kw)
                    acc += tile_input[(ty * shape.stride + kh) * SHARED_DIM + (tx * shape.stride + kw)] * tile_weight[kh * K + kw];
                }
            }
        }

        // SYNC: Ensure everyone is done reading before the next channel overwrites shared memory
        __syncthreads();
    }
    
    // 4. Write back
    if (oh < shape.out_height && ow < shape.out_width && oc < shape.filters) {
        output[output_index(shape, oc, oh, ow)] = acc;
    }
}

inline void launch_naive_conv2d(const float* d_input,
                                const float* d_weight,
                                float* d_output,
                                const Conv2dShape& shape,
                                cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_conv_grid(shape);
    conv2d_naive_kernel<<<grid, block, 0, stream>>>(d_input, d_weight, d_output, shape);
    /* TODO(student): check cudaGetLastError() and optionally cudaDeviceSynchronize() when debugging. */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

inline void launch_tiled_conv2d(const float* d_input,
                                const float* d_weight,
                                float* d_output,
                                const Conv2dShape& shape,
                                cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_conv_grid(shape);
    const int SHARED_DIM = BLOCK_SIZE + shape.kernel - 1;
    //size_t shared_bytes = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    size_t shared_bytes = (SHARED_DIM * SHARED_DIM + shape.kernel * shape.kernel) * sizeof(float);
    conv2d_tiled_kernel<<<grid, block, shared_bytes, stream>>>(d_input, d_weight, d_output, shape);
    /* TODO(student): choose a better shared-memory layout/size expression once kernels are implemented. */
    (void)d_input;
    (void)d_weight;
    (void)d_output;
}

inline double conv_gflops(const Conv2dShape& shape, double millis) {
    const double flops = static_cast<double>(shape.filters) * shape.out_height * shape.out_width *
                         shape.channels * shape.kernel * shape.kernel * 2.0;
    return flops / (millis * 1e6);
}
