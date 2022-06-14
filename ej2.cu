#include "./common.h"

__global__ void generator(int num_points, double *points) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < num_points && j < num_points) {
    points[i * num_points + j] = i + j;
  }
}

/*
  special_sum calculates the sum of all values inside a matrix in a radius.

  num_points: number of points in a matrix
  sum_result: pointer to the result array
  radius: radius of elements to sum
  matrix: pointer to the matrix with values
*/
__global__ void special_sum(unsigned int num_points, double *sum_result, int radius, double *matrix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ double matrix_point[32][32];

  if (i >= num_points || j >= num_points) {
    return;
  }

  int threadIdx_x = threadIdx.x + radius;
  int threadIdx_y = threadIdx.y + radius;

  matrix_point[threadIdx_y -radius][threadIdx_x -radius] = matrix[i * num_points + j];
  __syncthreads();

  double result = -2 * radius * matrix_point[threadIdx_y][threadIdx_x];

  for (int offset = -radius; offset <= radius; offset++) {
    if (threadIdx_x + offset >= 0 && threadIdx_x + offset < num_points) {
      result += matrix_point[threadIdx_y][threadIdx_x + offset];
    }
    if (threadIdx_y + offset >= 0 && threadIdx_y + offset < num_points) {
      result += matrix_point[threadIdx_y + offset][threadIdx_x];
    }
    // result += matrix_point[threadIdx_y + offset][threadIdx_x];
    // result += matrix_point[threadIdx_y][threadIdx_x + offset];
  }

  sum_result[j + i * num_points] = result / SMALL_POINT_SIZE;
}

__global__ void special_sum_org(unsigned int num_points, double *sum_result, int radius, double *matrix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= num_points || j >= num_points) {
    return;
  }

  double result = -2 * radius * matrix[i * num_points + j];

  for (int offset = -radius; offset <= radius; offset++) {
    if (i + offset >= 0 && i + offset < num_points) {
      result += matrix[(i + offset) * num_points + j];
    }
    if (j + offset >= 0 && j + offset < num_points) {
      result += matrix[i * num_points + j + offset];
    }
  }

  sum_result[j + i * num_points] = result / SMALL_POINT_SIZE;
}

int main(int argc, char *argv[]) {
	int num_points_2d = 0;
  int variant = 0;

	if (argc < 3) {
		printf("Debe ingresar la cantidad de puntos\n");
		return 0;
	} else {
		num_points_2d = atoi(argv[1]);
    variant = atoi(argv[2]);
    printf("Launched with %d\n", num_points_2d);
	}

  size_t size_2d = num_points_2d * num_points_2d * sizeof(double);

  double *d_points_2d;
  CUDA_CHK(cudaMalloc((void**)&d_points_2d, size_2d));

  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid_dim(num_points_2d / BLOCK_SIZE, num_points_2d / BLOCK_SIZE);

  // printf("block: %d x %d, grid: %d x %d\n", block_dim.x, block_dim.y, grid_dim.x, grid_dim.y);

  // Generates points
  generator<<<grid_dim, block_dim>>>(num_points_2d, d_points_2d);
  CUDA_CHK(cudaGetLastError());
  // CUDA_CHK(cudaDeviceSynchronize());

  double *gpu_special_sum_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_special_sum_result, size_2d));

  if (variant == 0) {
    special_sum_org<<<grid_dim, block_dim>>>(num_points_2d, gpu_special_sum_result, 2, d_points_2d);
  } else {
    special_sum<<<grid_dim, block_dim>>>(num_points_2d, gpu_special_sum_result, 2, d_points_2d);
  }
  CUDA_CHK(cudaGetLastError());
  // CUDA_CHK(cudaDeviceSynchronize());

  // double *special_sum_result = (double *)malloc(size_2d);
  // CUDA_CHK(cudaMemcpy(special_sum_result, gpu_special_sum_result, size_2d, cudaMemcpyDeviceToHost));

  // print_matrix_of_points(special_sum_result, 64);

  // free(special_sum_result);
  CUDA_CHK(cudaFree(d_points_2d));
  CUDA_CHK(cudaFree(gpu_special_sum_result));

  return 0;
}
