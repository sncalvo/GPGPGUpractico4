
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

__global__ void generator(int num_points, int *points) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < num_points) {
		// Fills with random int
    points[i] = i;
  }
}


// se asume que el tamaño de perm es igual al del bloque
// y que las premutaciones son válidas
__global__ void block_perm(int *data, int *perm, int length) {
	int off = blockIdx.x * blockDim.x;
	__shared__ int shared_pem[256];
	__shared__ int perm_data[256];

	if (length < off + threadIdx.x) return;

	shared_pem[threadIdx.x] = perm[threadIdx.x];
	__syncwarp();
	perm_data[threadIdx.x] = data[off + shared_pem[threadIdx.x]];
	__syncthreads();

	data[off+threadIdx.x] = perm_data[threadIdx.x];
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
	cudaMalloc(&perm, sizeof(int) * 256);

	cudaMemset(data, 0, sizeof(int) * length);

	// Fill perm with random int
	generator<<<1, 256>>>(256, perm);

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
