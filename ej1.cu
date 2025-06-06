#include <stdio.h>
#include <stdlib.h>

#include <locale.h>

#include "cuda.h"

#define M 256
#define BLOCK_SIZE 128
#define BLOCK_PROCESS_SIZE 32

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

#define N 512

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
int get_text_length(const char * fname);

__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void decrypt_kernel(int *d_message, int length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < length)
	{
		d_message[i] = modulo(A_MMI_M * (d_message[i] - B), M);
	}
}

__global__ void shared_count_occurences(int *d_message, int occurenses[M], int length, int process_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int block = i * process_size;

	extern __shared__ int shared_message[]; // blockDim * BLOCK_PROCESS_SIZE * sizeof(int) bytes
	__shared__ int local_occurenses[M]; // blockDim * sizeof(int) bytes

	for (int j = 0; j < process_size; j++)
	{
		if (j >= length) {
			break;
		}

		int index = threadIdx.x * process_size + j;
		shared_message[index] = d_message[block + j];
		__syncwarp();
		int char_index = modulo(shared_message[index], M);
		atomicAdd(&local_occurenses[char_index], 1);
	}
	__syncthreads();

	for (int j = 0; j < M; j++)
	{
		atomicAdd(&occurenses[j], local_occurenses[j]);
		__syncthreads();
	}
}

__global__ void count_occurences(int *d_message, int occurenses[M], int length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int occurense_index = modulo(d_message[i], M);

	if (i < length)
	{
		atomicAdd(&occurenses[occurense_index], 1);
		// occurenses[modulo(d_message[i], M)]++;
		// __syncthreads();
	}
}

int parte_2(int length, unsigned int size, int *message, int *occurenses, int block_amount, int variant)
{
	int *d_message;
	cudaMalloc((void**)&d_message, length * sizeof(int));
	cudaMemcpy(d_message, message, length * sizeof(int), cudaMemcpyHostToDevice);

	int *d_occurenses;
	cudaMalloc((void**)&d_occurenses, M * sizeof(int));
	cudaMemset(d_occurenses, 0, M * sizeof(int));

	dim3 block_dim(256);
 	dim3 grid_dim(size / block_dim.x);

	decrypt_kernel<<<grid_dim, block_dim>>>(d_message, length);
	if (variant == 1)
	{
		int process_size = length / (grid_dim.x * block_dim.x);
		block_dim = dim3(M);
		grid_dim = dim3(block_amount);
		shared_count_occurences<<<grid_dim, block_dim, M * process_size * sizeof(int)>>>(d_message, d_occurenses, length, process_size);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
	}
	else
	{
		count_occurences<<<grid_dim, block_dim>>>(d_message, d_occurenses, length);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
	}
	// count_occurences<<<grid_dim, block_dim, BLOCK_SIZE * sizeof(int)>>>(d_message, d_occurenses, length);

	cudaMemcpy(message, d_message, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(occurenses, d_occurenses, M * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_message);
	cudaFree(d_occurenses);

	return 0;
}

void print_occurences(int *occurenses)
{
	for (int i = 0; i < 256; i++)
	{
		if (occurenses[i] > 0)
		{
			printf("%d: %d\n", i, occurenses[i]);
		}
	}
}

void print_message(int *message, int length)
{
	for (int i = 0; i < 2048; i++)
	{
		printf("%c", (char)message[i]);
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	int *h_message;
	// int *d_message;
	unsigned int size;

	const char *fname;
	int block_amount = 0;
	int variant = 0;

	if (argc < 4) {
		printf("Debe ingresar el nombre del archivo\n");
		return 0;
	} else {
		fname = argv[1];
		block_amount = atoi(argv[2]);
		variant = atoi(argv[3]);
	}

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	int *h_occurenses = (int *)malloc(M * sizeof(int));

	parte_2(length, size, h_message, h_occurenses, block_amount, variant);

	free(h_occurenses);

	return 0;
}

int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);
	fseek(f, 0, SEEK_END);
	size_t length = ftell(f);
	fseek(f, pos, SEEK_SET);

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL) {
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c;
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}
