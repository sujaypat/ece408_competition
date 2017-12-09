
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
#define TILE_SIZE 4
// #define TILE_WIDTH 4


__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  unsigned row, col;
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];
  float sum = 0.0;
  
  row = blockIdx.y*blockDim.y + threadIdx.y;
  col = blockIdx.x*blockDim.x + threadIdx.x;

	for(int m = 0; m < (TILE_SIZE+numAColumns-1)/TILE_SIZE+1; m++)
	{
		if(row < numARows && m*TILE_SIZE+threadIdx.x < numAColumns)
		{
      		tileA[threadIdx.y][threadIdx.x] = A[row*numAColumns + m*TILE_SIZE+threadIdx.x];
    	}
    	else
    	{
			tileA[threadIdx.y][threadIdx.x] = 0.0;
    	}

		if(m*TILE_SIZE+threadIdx.y < numBRows && col < numBColumns)
		{
			tileB[threadIdx.y][threadIdx.x] = B[(m*TILE_SIZE+threadIdx.y)*numBColumns + col];
		}
		else
		{
			tileB[threadIdx.y][threadIdx.x] = 0.0;
		}
		__syncthreads();

		if(row < numCRows && col < numCColumns)
		{
				for(int k = 0; k < TILE_SIZE; k++)
				{
					sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
				}
		}
		__syncthreads();
	}
	if(row < numCRows && col < numCColumns)
	{
			C[row*numCColumns+col] = sum;
	}
}

template<typename gpu, typename DType>
__global__ void unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll)
{
	int c, s, h_out, w_out, h_unroll, w_base, p, q;
	int t = blockIdx.x * CUDA MAX_NUM_THREADS + threadIdx.x;
	int h_out = H – K + 1;
	int w_out = W – K + 1;
	int w_unroll = H_out * W_out;
	if (t < C * w_unroll) {
		c = t / w_unroll;
		s = t % w_unroll;
		h_out = s / w_out;
		w_out = s % w_out;
		h_unroll = h_out * w_out + w_out;
		w_base = c * K * K;
		for(p = 0; p < K; p++) {
			for(q = 0; q < K; q++) {
				w_unroll = w_base + p * K + q;
				X_unroll(w_unroll, h_unroll) = X(c, h_out + p, w_out + q);
			}
		}
	}
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

	int H_out = H – K + 1;
	int W_out = W – K + 1;
	int num_threads = C * H_out * W_out;
	int num_blocks = ceil((C * H_out * W_out) / CUDA_MAX_NUM_THREADS);
	unroll_Kernel<<<num_blocks, CUDA_MAX_NUM_THREADS>>>(C, H, W, K , x, x_unroll);


	dim3 blockDim(ceil((numCColumns/(double)TILE_WIDTH)), ceil(numCRows/(double)TILE_WIDTH), 1);
	dim3 gridDim(TILE_SIZE, TILE_SIZE, 1);

	matrixMultiply<<<gridDim, blockDim>>>(k, x_unroll, y, );




	// // Set the kernel dimensions
	// dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
	// dim3 gridDim(B, M, Z);
	
	// dim3 unroll_blockDim(TILE_SIZE, TILE_SIZE, 1);
	// dim3 unroll_gridDim(B, C, Z);

	// // Call the kernel
	// forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, M, C, H, W, K, W_grid);
	// MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

#undef TILE_SIZE
}
}
#endif
