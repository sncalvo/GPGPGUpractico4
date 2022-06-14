
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

// se asume que el tamaño de perm es igual al del bloque
// y que las premutaciones son válidas
__global__ void block_perm(int *data, int *perm, int length) {
	int off = blockIdx.x * blockDim.x;
	__shared__ int *shared_pem;
	int perm_data;

	if (length < off + threadIdx.x) return;

	shared_pem = &perm[threadIdx.x];
	__syncwarp();
	perm_data = data[off + *shared_pem];
	__syncthreads();

	data[off+threadIdx.x] = perm_data;
}

__global__ void block_perm_org(int * data, int *perm, int length) {
	int off = blockIdx.x * blockDim.x;

	if (length < off+threadIdx.x) return;

	int perm_data = data[off + perm[threadIdx.x]];

	__syncthreads();

	data[off + threadIdx.x] = perm_data;
}

int main(int argc, char *argv[]) {
	int *data, *perm;

	if (argc < 3) {
		printf("Usage: %s <data_length> <variant>\n", argv[0]);
		return 1;
	}

	int length = atoi(argv[1]);
	length = length * length;
	int variant = atoi(argv[2]);

	cudaMalloc(&data, sizeof(int) * length);
	cudaMalloc(&perm, sizeof(int) * length);

	cudaMemset(data, 0, sizeof(int) * length);
	cudaMemset(perm, 0, sizeof(int) * length);

	dim3 dimBlock(256, 1, 1);
	dim3 dimGrid(length / 256, 1, 1);

	if (variant == 0) {
		block_perm_org<<<dimGrid, dimBlock>>>(data, perm, length);
	} else {
		block_perm<<<dimGrid, dimBlock>>>(data, perm, length);
	}

	cudaFree(data);
	cudaFree(perm);
}
