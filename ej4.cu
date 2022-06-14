#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#define TSZ 32
#define DATA_SIZE 64

__global__ void sum_col_block(int *data, int length) {
	__shared__ int sh_tile[TSZ][TSZ];

	int n = gridDim.x * blockDim.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	sh_tile[threadIdx.y][threadIdx.x] = data[idx*n+idy];

	__syncthreads();

	int col_sum = sh_tile[threadIdx.y][threadIdx.x];

	for (int i=16; i>0; i/=2)
		col_sum += __shfl_down_sync(0xFFFFFFFF, col_sum, i);

	data[idy*n+idx] = col_sum;
}

__global__ void sum_col_block_opt(int *data, int length) {
	__shared__ int sh_tile[TSZ][TSZ];

	int n = gridDim.x * blockDim.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	sh_tile[threadIdx.y][threadIdx.x] = data[idy*n+idx];

	__syncwarp();

	int col_sum = sh_tile[threadIdx.x][threadIdx.y];

	// for (int i=16; i>0; i/=2)
	// {
	// 	if (threadIdx.x + i < TSZ)
	// 		col_sum += sh_tile[threadIdx.x][threadIdx.y+i];
	// }
	for (int i=16; i>0; i/=2)
	{
		if (threadIdx.x + i < TSZ)
			col_sum += sh_tile[threadIdx.x][threadIdx.y+i];
	}

	data[idy*n+idx] = col_sum;
}

int main() {
	int *data_host = (int*)malloc(sizeof(int)*DATA_SIZE*DATA_SIZE);
	int *data;
	cudaMalloc((void **)&data, sizeof(int)*DATA_SIZE*DATA_SIZE);

	for (int i=0; i<DATA_SIZE; i++)
		for (int j=0; j<DATA_SIZE; j++)
			data_host[i*DATA_SIZE+j] = j;

	printf("IN: \n");
	for (int i=0; i<DATA_SIZE; i++) {
		for (int j=0; j<DATA_SIZE; j++)
			printf("%d ", data_host[i*DATA_SIZE+j]);

		printf("\n");
	}
	printf("\n");

	printf("================\n");

	cudaMemcpy(data, data_host, sizeof(int)*DATA_SIZE*DATA_SIZE, cudaMemcpyHostToDevice);

	dim3 dimBlock(TSZ, TSZ);
	dim3 dimGrid(DATA_SIZE/TSZ, DATA_SIZE/TSZ);

	sum_col_block<<<dimGrid, dimBlock>>>(data, DATA_SIZE*DATA_SIZE);
	cudaDeviceSynchronize();
	cudaMemcpy(data_host, data, sizeof(int)*DATA_SIZE*DATA_SIZE, cudaMemcpyDeviceToHost);

	printf("RES: \n");
	for (int j=0; j<DATA_SIZE; j++) {
		for (int i=0; i<DATA_SIZE; i++)
			printf("%d ", data_host[i*DATA_SIZE+j]);

		printf("\n");
	}
	printf("\n");

	cudaFree(data);

	free(data_host);

	return 0;
}
