#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#define TSZ 32
__global__ void sum_col_block(int *data, int length) {
	__shared__ int sh_tile[TSZ][TSZ];

	int n = gridDim.x * blockDim.x;
	int idx = blockIdx.x * blockDim.x+threadIdx.x;
	int idy = blockIdx.y * blockDim.y+threadIdx.y;

	sh_tile[threadIdx.y][threadIdx.x] = data[idy*n+idx];

	__syncthreads();

	int col_sum = sh_tile[threadIdx.x][threadIdx.y];

	for (int i=16; i>0; i/=2)
		col_sum += __shfl_down_sync(0xFFFFFFFF, col_sum, i);

	data[idy*n+idx] = col_sum;
}

int main() {
	int *data;
	cudaMalloc(&data, sizeof(int)*TSZ*TSZ);

	for (int i=0; i<TSZ*TSZ; i++)
		data[i]=i;

	sum_col_block<<<TSZ/32, TSZ/32>>>(data, TSZ);

	int *data_host;
	cudaMemcpy(&data_host, data, sizeof(int)*TSZ*TSZ, cudaMemcpyDeviceToHost);

	for (int i=0; i<TSZ*TSZ; i++)
		printf("%d ", data[i]);

	cudaFree(data);

	free(data_host);

	return 0;
}
