
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 28

namespace mxnet
{
namespace op
{

__constant__ float constant_kernel[1250];

template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    __shared__ DType x_shared[28 * 28];

    int block_num = blockIdx.x;
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    int input_size = 28 * 28;
    int output_size = 24 * 24;
    int read_base = block_num * input_size + local_h * 28 + local_w;
    int write_base = 0;
    DType sum = 0.0;
    // load 28 * 28 input
    x_shared[local_h * 28 + local_w] = x[read_base];
    __syncthreads();

    // compute
    if (local_h < 24 && local_w < 24) {
        write_base = block_num * 50 * output_size + local_h * 24 + local_w;
        #pragma unroll 3
        for (int kernel_index = 0; kernel_index < 50; ++kernel_index) {
            sum = 0.0;
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                    sum += x_shared[(local_h + i) * 28 + local_w + j] *
                    constant_kernel[kernel_index * 25 + i * 5 + j];
                }
            }
            y[write_base + kernel_index * output_size] = sum;
        }
        write_base += 50 * output_size;
    }
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {


    // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) << "Starting a GPU implementation based on share memory!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; // input batch
    const int M = w.shape_[0]; // output channel number
    const int C = x.shape_[1]; // input channel number
    const int H = x.shape_[2]; // input height
    const int W = x.shape_[3]; // input width
    const int K = w.shape_[2]; // kernel size

    // Set the kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, 1, 1);

    // allocate constant_kernel
    cudaMemcpyToSymbol(constant_kernel, w.dptr_, sizeof(float) * 50 * 25, 0, cudaMemcpyDeviceToDevice);
    // Call the kernel                                0 is sharemem s is stream
    forward_kernel<gpu, DType><<<gridDim, blockDim, sizeof(DType) * H * W, s>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
