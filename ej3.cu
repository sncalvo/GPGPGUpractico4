#include "cuda.h"

// se asume que el tamaño de perm es igual al del bloque
// y que las premutaciones son válidas
__global__ void block_perm(int * data, int *perm, int length){
	int off = blockIdx.x * blockDim.x;
	
	if (length < off + threadIdx.x) return;

	__shared__ int *shared_pem = perm[threadIdx.x];
	__shared__ int *perm_data = data[off + perm[threadIdx.x]];
	__syncwarp();

	data[off+threadIdx.x] = perm_data;
}
