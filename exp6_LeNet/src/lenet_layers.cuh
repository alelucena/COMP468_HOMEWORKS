#pragma once

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#include <array>
#include <string>
#include <vector>

struct LenetShape {
    int batch;
    // Assume MNIST-style 1x32x32 input.
    static constexpr int in_channels = 1;
    static constexpr int in_height = 32;
    static constexpr int in_width = 32;

    static constexpr int conv1_out_channels = 6;
    static constexpr int conv1_kernel = 5;
    static constexpr int conv2_out_channels = 16;
    static constexpr int conv2_kernel = 5;

    static constexpr int pool_stride = 2;

    static constexpr int fc1_out = 120;
    static constexpr int fc2_out = 84;
    static constexpr int fc3_out = 10;

    size_t input_elements;
    size_t conv1_out_elems;
    size_t pool1_out_elems;
    size_t conv2_out_elems;
    size_t pool2_out_elems;
    size_t fc1_out_elems;
    size_t fc2_out_elems;
    size_t output_elements;

    size_t total_weight_elements;
    size_t total_bias_elements;
    std::vector<size_t> weight_offsets;
    std::vector<size_t> bias_offsets;
};

inline LenetShape make_lenet_shape(int batch) {
    LenetShape s{};
    s.batch = batch;
    const int conv1_out_h = LenetShape::in_height - LenetShape::conv1_kernel + 1;  // stride=1, padding=0
    const int conv1_out_w = LenetShape::in_width - LenetShape::conv1_kernel + 1;
    const int pool1_out_h = conv1_out_h / LenetShape::pool_stride;
    const int pool1_out_w = conv1_out_w / LenetShape::pool_stride;

    const int conv2_in_h = pool1_out_h;
    const int conv2_in_w = pool1_out_w;
    const int conv2_out_h = conv2_in_h - LenetShape::conv2_kernel + 1;
    const int conv2_out_w = conv2_in_w - LenetShape::conv2_kernel + 1;
    const int pool2_out_h = conv2_out_h / LenetShape::pool_stride;
    const int pool2_out_w = conv2_out_w / LenetShape::pool_stride;

    const int flattened = LenetShape::conv2_out_channels * pool2_out_h * pool2_out_w;

    s.input_elements = static_cast<size_t>(batch) * LenetShape::in_channels * LenetShape::in_height * LenetShape::in_width;
    s.conv1_out_elems = static_cast<size_t>(batch) * LenetShape::conv1_out_channels * conv1_out_h * conv1_out_w;
    s.pool1_out_elems = static_cast<size_t>(batch) * LenetShape::conv1_out_channels * pool1_out_h * pool1_out_w;
    s.conv2_out_elems = static_cast<size_t>(batch) * LenetShape::conv2_out_channels * conv2_out_h * conv2_out_w;
    s.pool2_out_elems = static_cast<size_t>(batch) * LenetShape::conv2_out_channels * pool2_out_h * pool2_out_w;
    s.fc1_out_elems = static_cast<size_t>(batch) * LenetShape::fc1_out;
    s.fc2_out_elems = static_cast<size_t>(batch) * LenetShape::fc2_out;
    s.output_elements = static_cast<size_t>(batch) * LenetShape::fc3_out;

    s.weight_offsets = std::vector<size_t>(5, 0);
    s.bias_offsets = std::vector<size_t>(5, 0);
    size_t cursor_w = 0;
    size_t cursor_b = 0;
    const auto push = [&](size_t elements, std::vector<size_t>& offsets, size_t& cursor) {
        offsets.push_back(cursor);
        cursor += elements;
    };
    s.weight_offsets[0] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::conv1_out_channels) * LenetShape::in_channels * LenetShape::conv1_kernel * LenetShape::conv1_kernel;
    s.weight_offsets[1] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::conv2_out_channels) * LenetShape::conv1_out_channels * LenetShape::conv2_kernel * LenetShape::conv2_kernel;
    s.weight_offsets[2] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::fc1_out) * flattened;
    s.weight_offsets[3] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::fc2_out) * LenetShape::fc1_out;
    s.weight_offsets[4] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::fc3_out) * LenetShape::fc2_out;

    s.bias_offsets[0] = cursor_b;
    cursor_b += LenetShape::conv1_out_channels;
    s.bias_offsets[1] = cursor_b;
    cursor_b += LenetShape::conv2_out_channels;
    s.bias_offsets[2] = cursor_b;
    cursor_b += LenetShape::fc1_out;
    s.bias_offsets[3] = cursor_b;
    cursor_b += LenetShape::fc2_out;
    s.bias_offsets[4] = cursor_b;
    cursor_b += LenetShape::fc3_out;

    s.total_weight_elements = cursor_w;
    s.total_bias_elements = cursor_b;
    return s;
}

inline double lenet_gflops(const LenetShape& shape, double millis) {
    const double conv1_flops = static_cast<double>(shape.batch) * LenetShape::conv1_out_channels * LenetShape::in_channels *
                               LenetShape::conv1_kernel * LenetShape::conv1_kernel * 2.0 * 28 * 28;
    const double conv2_flops = static_cast<double>(shape.batch) * LenetShape::conv2_out_channels * LenetShape::conv1_out_channels *
                               LenetShape::conv2_kernel * LenetShape::conv2_kernel * 2.0 * 10 * 10;
    const double fc1_in = LenetShape::conv2_out_channels * 5 * 5;
    const double fc_flops = static_cast<double>(shape.batch) *
                            (2.0 * fc1_in * LenetShape::fc1_out +
                             2.0 * LenetShape::fc1_out * LenetShape::fc2_out +
                             2.0 * LenetShape::fc2_out * LenetShape::fc3_out);
    const double total = conv1_flops + conv2_flops + fc_flops;
    return total / (millis * 1e6);
}

struct LenetDescriptors {
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnTensorDescriptor_t conv1_out_desc = nullptr;
    cudnnTensorDescriptor_t pool1_out_desc = nullptr;
    cudnnTensorDescriptor_t conv2_out_desc = nullptr;
    cudnnTensorDescriptor_t pool2_out_desc = nullptr;
    cudnnTensorDescriptor_t fc1_desc = nullptr;
    cudnnTensorDescriptor_t fc2_desc = nullptr;
    cudnnTensorDescriptor_t fc3_desc = nullptr;

    cudnnFilterDescriptor_t conv1_filter = nullptr;
    cudnnFilterDescriptor_t conv2_filter = nullptr;

    cudnnConvolutionDescriptor_t conv1_desc = nullptr;
    cudnnConvolutionDescriptor_t conv2_desc = nullptr;

    cudnnActivationDescriptor_t activation = nullptr;
    cudnnPoolingDescriptor_t pool = nullptr;
};

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

void check_cudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : " + cudnnGetErrorString(status));
    }
}

inline void create_lenet_descriptors(const LenetShape& shape, LenetDescriptors& d) {
    /* TODO(student): cudnnCreate* all descriptors and configure tensor dimensions/strides. */

    // --- 1. Create all descriptors (Initializing the handles) ---
    check_cudnn(cudnnCreateTensorDescriptor(&d.input_desc), "create input_desc");
    check_cudnn(cudnnCreateTensorDescriptor(&d.conv1_out_desc), "create conv1_out_desc");
    check_cudnn(cudnnCreateTensorDescriptor(&d.pool1_out_desc), "create pool1_out_desc");
    check_cudnn(cudnnCreateTensorDescriptor(&d.conv2_out_desc), "create conv2_out_desc");
    check_cudnn(cudnnCreateTensorDescriptor(&d.pool2_out_desc), "create pool2_out_desc");
    
    // Treat FC descriptors as 4D tensors with 1x1 spatial dimensions
    check_cudnn(cudnnCreateTensorDescriptor(&d.fc1_desc), "create fc1_desc");
    check_cudnn(cudnnCreateTensorDescriptor(&d.fc2_desc), "create fc2_desc");
    check_cudnn(cudnnCreateTensorDescriptor(&d.fc3_desc), "create fc3_desc");

    check_cudnn(cudnnCreateFilterDescriptor(&d.conv1_filter), "create conv1_filter");
    check_cudnn(cudnnCreateFilterDescriptor(&d.conv2_filter), "create conv2_filter");

    check_cudnn(cudnnCreateConvolutionDescriptor(&d.conv1_desc), "create conv1_desc");
    check_cudnn(cudnnCreateConvolutionDescriptor(&d.conv2_desc), "create conv2_desc");

    check_cudnn(cudnnCreateActivationDescriptor(&d.activation), "create activation");
    check_cudnn(cudnnCreatePoolingDescriptor(&d.pool), "create pool");

    // --- 2. Configure Convolutional Layer 1 (C1) ---
    // Input: Batch x 1 x 32 x 32
    check_cudnn(cudnnSetTensor4dDescriptor(d.input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           shape.batch, 1, 32, 32), "set input_desc");
    // Filter: 6 x 1 x 5 x 5
    check_cudnn(cudnnSetFilter4dDescriptor(d.conv1_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 
                                           6, 1, 5, 5), "set conv1_filter");
    // Conv: Padding 0, Stride 1
    check_cudnn(cudnnSetConvolution2dDescriptor(d.conv1_desc, 0, 0, 1, 1, 1, 1, 
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "set conv1_desc");
    // Output: Batch x 6 x 28 x 28
    check_cudnn(cudnnSetTensor4dDescriptor(d.conv1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           shape.batch, 6, 28, 28), "set conv1_out_desc");

    // --- 3. Configure Pooling Layer 1 (S2) ---
    // Average Pooling: 2x2 window, Stride 2
    check_cudnn(cudnnSetPooling2dDescriptor(d.pool, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 
                                            CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2), "set pool");
    // Output: Batch x 6 x 14 x 14
    check_cudnn(cudnnSetTensor4dDescriptor(d.pool1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           shape.batch, 6, 14, 14), "set pool1_out_desc");

    // --- 4. Configure Convolutional Layer 2 (C3) ---
    // Filter: 16 x 6 x 5 x 5
    check_cudnn(cudnnSetFilter4dDescriptor(d.conv2_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 
                                           16, 6, 5, 5), "set conv2_filter");
    // Conv: Padding 0, Stride 1 (reuse settings from conv1_desc if preferred, but explicit here)
    check_cudnn(cudnnSetConvolution2dDescriptor(d.conv2_desc, 0, 0, 1, 1, 1, 1, 
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "set conv2_desc");
    // Output: Batch x 16 x 10 x 10
    check_cudnn(cudnnSetTensor4dDescriptor(d.conv2_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           shape.batch, 16, 10, 10), "set conv2_out_desc");

    // --- 5. Configure Pooling Layer 2 (S4) ---
    // Output: Batch x 16 x 5 x 5
    check_cudnn(cudnnSetTensor4dDescriptor(d.pool2_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           shape.batch, 16, 5, 5), "set pool2_out_desc");

    // --- 6. Configure Fully Connected Layers (FC1, FC2, FC3) ---
    // We treat FC layers as 4D tensors where Height=1 and Width=1
    // FC1 Output: Batch x 120 x 1 x 1
    check_cudnn(cudnnSetTensor4dDescriptor(d.fc1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           shape.batch, 120, 1, 1), "set fc1_desc");
    // FC2 Output: Batch x 84 x 1 x 1
    check_cudnn(cudnnSetTensor4dDescriptor(d.fc2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           shape.batch, 84, 1, 1), "set fc2_desc");
    // FC3 (Final) Output: Batch x 10 x 1 x 1
    check_cudnn(cudnnSetTensor4dDescriptor(d.fc3_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           shape.batch, 10, 1, 1), "set fc3_desc");

    // --- 7. Configure Activation ---
    check_cudnn(cudnnSetActivationDescriptor(d.activation, CUDNN_ACTIVATION_TANH, 
                                             CUDNN_NOT_PROPAGATE_NAN, 0.0), "set activation");
}

inline void destroy_lenet_descriptors(LenetDescriptors& d) {
    /* TODO(student): destroy all descriptors created above. */

    // Destroy Tensor Descriptors
    if (d.input_desc)     check_cudnn(cudnnDestroyTensorDescriptor(d.input_desc), "destroy input_desc");
    if (d.conv1_out_desc) check_cudnn(cudnnDestroyTensorDescriptor(d.conv1_out_desc), "destroy conv1_out_desc");
    if (d.pool1_out_desc) check_cudnn(cudnnDestroyTensorDescriptor(d.pool1_out_desc), "destroy pool1_out_desc");
    if (d.conv2_out_desc) check_cudnn(cudnnDestroyTensorDescriptor(d.conv2_out_desc), "destroy conv2_out_desc");
    if (d.pool2_out_desc) check_cudnn(cudnnDestroyTensorDescriptor(d.pool2_out_desc), "destroy pool2_out_desc");
    if (d.fc1_desc)       check_cudnn(cudnnDestroyTensorDescriptor(d.fc1_desc), "destroy fc1_desc");
    if (d.fc2_desc)       check_cudnn(cudnnDestroyTensorDescriptor(d.fc2_desc), "destroy fc2_desc");
    if (d.fc3_desc)       check_cudnn(cudnnDestroyTensorDescriptor(d.fc3_desc), "destroy fc3_desc");

    // Destroy Filter Descriptors
    if (d.conv1_filter)   check_cudnn(cudnnDestroyFilterDescriptor(d.conv1_filter), "destroy conv1_filter");
    if (d.conv2_filter)   check_cudnn(cudnnDestroyFilterDescriptor(d.conv2_filter), "destroy conv2_filter");

    // Destroy Convolution Descriptors
    if (d.conv1_desc)     check_cudnn(cudnnDestroyConvolutionDescriptor(d.conv1_desc), "destroy conv1_desc");
    if (d.conv2_desc)     check_cudnn(cudnnDestroyConvolutionDescriptor(d.conv2_desc), "destroy conv2_desc");

    // Destroy Activation and Pooling Descriptors
    if (d.activation)     check_cudnn(cudnnDestroyActivationDescriptor(d.activation), "destroy activation");
    if (d.pool)           check_cudnn(cudnnDestroyPoolingDescriptor(d.pool), "destroy pool");

    // Set all to nullptr for safety
    d.input_desc = nullptr;
    d.conv1_out_desc = nullptr;
    d.pool1_out_desc = nullptr;
    d.conv2_out_desc = nullptr;
    d.pool2_out_desc = nullptr;
    d.fc1_desc = nullptr;
    d.fc2_desc = nullptr;
    d.fc3_desc = nullptr;
    d.conv1_filter = nullptr;
    d.conv2_filter = nullptr;
    d.conv1_desc = nullptr;
    d.conv2_desc = nullptr;
    d.activation = nullptr;
    d.pool = nullptr;
}

inline cudnnConvolutionFwdAlgo_t parse_algo(const std::string& name) {
    if (name == "implicit_gemm") return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    if (name == "implicit_precomp") return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    if (name == "fft") return CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    // TODO(student): extend with more options.
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
}

inline size_t query_conv_workspace(cudnnHandle_t handle,
                                   const LenetShape& shape,
                                   const LenetDescriptors& descs,
                                   cudnnConvolutionFwdAlgo_t algo,
                                   bool second_conv) {
    /* TODO(student): call cudnnGetConvolutionForwardWorkspaceSize for conv1/conv2. */
    
    size_t workspace_size = 0;

    if (!second_conv) {
        // Query for Conv1: Input -> Conv1 Filter -> Conv1 Output
        check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            descs.input_desc,
            descs.conv1_filter,
            descs.conv1_desc,     // Don't forget the convolution descriptor!
            descs.conv1_out_desc,
            algo,
            &workspace_size
        ), "query for conv1");
    } else {
        // Query for Conv2: Pool1 Output -> Conv2 Filter -> Conv2 Output
        check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            descs.pool1_out_desc, // Conv2's input is Pool1's output
            descs.conv2_filter,
            descs.conv2_desc,
            descs.conv2_out_desc,
            algo,
            &workspace_size
        ), "query for conv2");
    }

    return workspace_size;
}

inline void run_lenet_conv(cudnnHandle_t handle,
                           const LenetShape& shape,
                           const LenetDescriptors& descs,
                           const float* d_input,
                           const float* d_filter,
                           const float* d_bias,      
                           size_t bias_offset,       
                           float* d_output,
                           void* d_workspace,
                           size_t workspace_bytes,
                           const std::string& algo_name,
                           bool second_conv) {
    /* TODO(student): select descriptors (conv1 vs conv2), pick algo, and call cudnnConvolutionForward.
       After conv, optionally launch cudnnBiasAdd + cudnnActivationForward (tanh/ReLU). */
    float alpha = 1.0f;
    float beta_zero = 0.0f;
    float beta_one = 1.0f;
    cudnnConvolutionFwdAlgo_t algo_desc = parse_algo(algo_name);

    // 1. CONVOLUTION
    check_cudnn(cudnnConvolutionForward(
        handle, &alpha,
        second_conv ? descs.pool1_out_desc : descs.input_desc, d_input,
        second_conv ? descs.conv2_filter : descs.conv1_filter, d_filter,
        second_conv ? descs.conv2_desc : descs.conv1_desc,
        algo_desc, d_workspace, workspace_bytes,
        &beta_zero,
        second_conv ? descs.conv2_out_desc : descs.conv1_out_desc, d_output
    ), "Convolution");

    // 2. BIAS ADDITION
    // Need a temporary descriptor for the bias tensor (1 x C x 1 x 1)
    cudnnTensorDescriptor_t bias_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "create bias desc");
    
    int channels = second_conv ? shape.conv2_out_channels : shape.conv1_out_channels;
    // Bias is always 1 x Channels x 1 x 1 for NCHW broadcasting
    check_cudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                           1, channels, 1, 1), "set bias desc");

    check_cudnn(cudnnAddTensor(
        handle, &alpha,
        bias_desc, d_bias + bias_offset, // Source: The bias values
        &beta_one,
        second_conv ? descs.conv2_out_desc : descs.conv1_out_desc, d_output // Dest: Add to conv output
    ), "Add bias");

    check_cudnn(cudnnDestroyTensorDescriptor(bias_desc), "destroy bias desc");

    // 3. ACTIVATION
    check_cudnn(cudnnActivationForward(
        handle, descs.activation, 
        &alpha,
        second_conv ? descs.conv2_out_desc : descs.conv1_out_desc, d_output,
        &beta_zero,
        second_conv ? descs.conv2_out_desc : descs.conv1_out_desc, d_output
    ), "Activation");
}

inline void run_lenet_pool(cudnnHandle_t handle,
                           const LenetDescriptors& descs,
                           const float* d_input,
                           float* d_output,
                           bool second_pool) {
    /* TODO(student): use cudnnPoolingForward for pool1 or pool2. */
    (void)handle;
    (void)descs;
    (void)d_input;
    (void)d_output;
    (void)second_pool;
    
    float alpha = 1.0f;
    float beta_zero = 0.0f;
    check_cudnn(cudnnPoolingForward(
        handle, 
        descs.pool,
        &alpha,
        // If second_pool is false, use Layer 1 descriptors. 
        // If true, use Layer 2 descriptors.
        second_pool ? descs.conv2_out_desc : descs.conv1_out_desc,
        d_input,
        &beta_zero, // Use the variable you defined above
        second_pool ? descs.pool2_out_desc : descs.pool1_out_desc,
        d_output
    ), "Pooling");
}

__global__ void fc_bias_activation_kernel(float* data, const float* bias, int size, int batch_Size, bool apply_tanh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Bias is per-output-feature; use modulo for batch broadcasting
        int bias_idx = idx / batch_Size;
        float val = data[idx] + bias[bias_idx];
        
        if (apply_tanh) {
            // Tanh activation 
            data[idx] = tanhf(val); 
        } else {
            data[idx] = val;
        }
       
    }
}

inline void run_fc_layer(cublasHandle_t handle,
                         const LenetShape& shape,
                         int layer_idx,
                         const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         cudaStream_t stream) {
    /* TODO(student): implement row-major GEMM via cublasSgemm / cublasGemmEx + bias add + activation.
       layer_idx ∈ {0:fc1,1:fc2,2:fc3}; use shape metadata to determine dims. */
    int batch = shape.batch;
    int in_features, out_features;

    // Determine dimensions based on layer_idx
    if (layer_idx == 0) { // FC1
        in_features = 400; // 5x5x16 from Pool2
        out_features = 120;
    } else if (layer_idx == 1) { // FC2
        in_features = 120;
        out_features = 84;
    } else { // FC3
        in_features = 84;
        out_features = 10;
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    // 1. GEMM call
    // Want: Output[M, N] = Weight[M, K] * Input[K, N]
    // In cuBLAS (Column-Major), this is equivalent to:
    // RowMajor_Matrix_Mult(A, B) == ColumnMajor_Matrix_Mult(B, A)
    check_cublas(cublasSgemm(handle, 
                            CUBLAS_OP_N,   // Don't transpose B (Batch is columns)
                            CUBLAS_OP_N,   // Don't transpose A (Weights are rows)
                            batch,         // New 'm'
                            out_features,  // New 'n'
                            in_features,   // New 'k'
                            &alpha,
                            d_input, batch,         // lda = batch
                            d_weight, in_features,  // ldb = in_features
                            &beta,
                            d_output, batch));      // ldc = batch
    
    // 2. Launch Bias + Activation Kernel
    // Add a boolean flag to the kernel to decide whether to Tanh or not
    bool apply_tanh = (layer_idx < 2); // true for FC1 (0) and FC2 (1), false for FC3 (2)

    int total_elements = out_features * batch;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    fc_bias_activation_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_output, d_bias, total_elements, shape.batch, apply_tanh
    );
}

inline void reshape_conv_to_fc(const LenetShape& shape, const float* d_input, float* d_output, cudaStream_t stream) {
    /* TODO(student): implement or call cudaMemcpy to treat tensor as flattened (B, flattened).
       A simple kernel can copy/reshape pool2 output into row-major batches for GEMM. */
    (void)shape;
    (void)d_input;
    (void)d_output;
    (void)stream;
}
