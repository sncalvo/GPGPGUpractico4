#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#define TSZ 32
#define DATA_SIZE 128

__global__ void sum_col_block(int *data, int length) {
	__shared__ int sh_tile[TSZ][TSZ];

	int n = gridDim.x * blockDim.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	sh_tile[threadIdx.y][threadIdx.x] = data[idy*n+idx];

	__syncthreads();

	int col_sum = sh_tile[threadIdx.x][threadIdx.y];

	for (int i=16; i>0; i/=2)
		col_sum += __shfl_down_sync(0xFFFFFFFF, col_sum, i);

	data[idy*n+idx] = col_sum;
}

int main() {
	int *data;
	cudaMalloc((void **)&data, sizeof(int)*DATA_SIZE*DATA_SIZE);

	for (int i=0; i<DATA_SIZE*DATA_SIZE; i++)
		data[i]=i;

	dim3 dimBlock(TSZ, TSZ);
	dim3 dimGrid(DATA_SIZE/TSZ, DATA_SIZE/TSZ);

	printf("Has not failed 1 \n");
	fflush(stdout);

	sum_col_block<<<dimGrid, dimBlock>>>(data, DATA_SIZE*DATA_SIZE);
	cudaDeviceSynchronize();

	printf("Has not failed 2 \n");
	fflush(stdout);

	int *data_host = (int*)malloc(sizeof(int)*DATA_SIZE*DATA_SIZE);
	cudaMemcpy(data_host, data, sizeof(int)*DATA_SIZE*DATA_SIZE, cudaMemcpyDeviceToHost);

	printf("Has not failed 3 \n");
	fflush(stdout);

	for (int i=0; i<DATA_SIZE*DATA_SIZE; i++)
		printf("%d ", data[i]);

	cudaFree(data);

	free(data_host);

	return 0;
}
