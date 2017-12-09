
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
#define TILE_SIZE 32

template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int M, const int C, const int H, const int W, const int K, const int W_grid)
{
	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	#define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
	#define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
	#define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

	int b = blockIdx.x;
	int m = blockIdx.y;
	int h = blockIdx.z / W_grid + threadIdx.y;
	int w = blockIdx.z % W_grid + threadIdx.x;
	DType acc = 0;

	if(h < H_out && w < W_out)
	{
		for(int c = 0; c < C; c++)
		{
			for(int p = 0; p < K; p++)
			{
				for(int q = 0; q < K; q++)
				{
					acc += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);    				
				}
			}
		}
		y4d(b, m, h, w) = acc;
	}

	#undef y4d
	#undef x4d
	#undef k4d
}


template<typename gpu, typename DType>
__global__ void unroll_kernel(const int B, const int C, const int H, const int W, const int K, const DType *x, DType *x_unroll)
{
	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	#define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
	#define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
	#define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

	int b = blockIdx.x;
	int c = blockIdx.y;
	int h = blockIdx.z / W_grid + threadIdx.y;
	int w = blockIdx.z % W_grid + threadIdx.x;
	DType acc = 0;

	int w_base = c * (K*K);
	for(int p = 0; p < K; p++)
	{
		for(int q = 0; q < K; q++)
		{
			x_unroll[b, h * W_out + w, w_base + p * K + q] = x[b, c, h + p, w + q];
		}

	}



/*
	if(h < H_out && w < W_out)
	{
		for(int c = 0; c < C; c++)
		{
			for(int p = 0; p < K; p++)
			{
				for(int q = 0; q < K; q++)
				{
					acc += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);    				
				}
			}
		}
		y4d(b, m, h, w) = acc;
	}
*/
	#undef y4d
	#undef x4d
	#undef k4d
}


// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
	// You'll probably need to launch kernels against the right stream to keep MXNet happy
	cudaStream_t s = y.stream_->stream_;

	// Extract the tensor dimensions into B,M,C,H,W,K
	const int B = x.shape_[0];
	const int M = y.shape_[1];
	const int C = x.shape_[1];
	const int H = x.shape_[2];
	const int W = x.shape_[3];
	const int K = w.shape_[3];

	const int W_grid = (int)ceil((W - K + 1) / ((float)TILE_SIZE));
	const int H_grid = (int)ceil((H - K + 1) / ((float)TILE_SIZE));
	const int Z = W_grid * H_grid;

	// Set the kernel dimensions
	dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
	dim3 gridDim(B, M, Z);

	// Call the kernel
	forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, M, C, H, W, K, W_grid);
	MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

#undef TILE_SIZE
}
}
#endif
