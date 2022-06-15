
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

// se asume que el tamaño de perm es igual al del bloque
// y que las premutaciones son válidas
__global__ void block_perm(int *data, int *perm, int length) {
	int off = blockIdx.x * blockDim.x;
	__shared__ int shared_data[1024];

	if (length < off + threadIdx.x) return;

	shared_data[threadIdx.x] = data[off];
	__syncthreads();

	data[off+threadIdx.x] = shared_data[perm[threadIdx.x]];
}

__global__ void block_perm_org(int * data, int *perm, int length) {
	int off = blockIdx.x * blockDim.x;

	if (length < off+threadIdx.x) return;

	int perm_data = data[off + perm[threadIdx.x]];

	__syncthreads();

	data[off + threadIdx.x] = perm_data;
}

int main(int argc, char *argv[]) {
	int *data, *perm, *d_perm;

	if (argc < 3) {
		printf("Usage: %s <data_length> <variant>\n", argv[0]);
		return 1;
	}

	perm = (int *)malloc(sizeof(int) * 1024);
	// generate random number between 0 and 1023
	for (int i = 0; i < 1024; i++) {
		perm[i] = rand() % 1024;
	}

	int length = atoi(argv[1]);
	length = length * length;
	int variant = atoi(argv[2]);

	cudaMalloc(&data, sizeof(int) * length);
	cudaMalloc(&perm, sizeof(int) * 1024);

	cudaMemset(data, 0, sizeof(int) * length);
	// copy perm to device d_perm
	cudaMemcpy(d_perm, perm, sizeof(int) * 1024, cudaMemcpyHostToDevice);

	// Fill perm with random int
	// generator<<<1, 1024>>>(1024, perm, length);

	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(length / 1024, 1, 1);

	if (variant == 0) {
		block_perm_org<<<dimGrid, dimBlock>>>(data, d_perm, length);
	} else {
		block_perm<<<dimGrid, dimBlock>>>(data, d_perm, length);
	}

	cudaFree(data);
	cudaFree(d_perm);

	free(perm);
}
