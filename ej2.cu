#include "./common.h"
#include "./generator.cuh"

__global__ void generator(unsigned int num_points, double *points) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

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

  unsigned int threadIdx_x = threadIdx.x + 2;
  unsigned int threadIdx_y = threadIdx.y + 2;

  matrix_point[threadIdx_y -1][threadIdx_x -1] = matrix[i * num_points + j];
  __syncwarp();

  double result = -2 * radius * matrix_point[threadIdx_y][threadIdx_x];

  for (int offset = -radius; offset <= radius; offset++) {
    result += matrix_point[threadIdx_y + offset][threadIdx_x];
    result += matrix_point[threadIdx_y][threadIdx_x + offset];
  }

  sum_result[j + i * num_points] = result / SMALL_POINT_SIZE;
}


// calculate_sin: Calculates sin of x and y coordinates for points inside a square
__global__ void calculate_sin(unsigned int num_points, Point2D *points, double *sin_result) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point2D *point;
  point = &points[j + i * num_points];

  double x = point->x;
  double y = point->y;

  sin_result[j + i * num_points] = sin(x + y);
}

int main(int argc, char *argv[]) {
	int num_points_2d = 0;

	if (argc < 2) {
		printf("Debe ingresar la cantidad de puntos\n");
		return 0;
	} else {
		num_points_2d = atoi(argv[1]);
	}

  // Builds square of points with space of SMALL_POINT_SIZE
  // unsigned int num_points_2d = trunc(SQUARE_LENGTH / SMALL_POINT_SIZE); // 6_280

  size_t size_2d = num_points_2d * num_points_2d * sizeof(double); // 39_438_400 x double_size

  double *d_points_2d;
  CUDA_CHK(cudaMalloc((void**)&d_points_2d, size_2d));

  // dim3 block_dim(32, 32, 1);
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid_dim(num_points_2d / BLOCK_SIZE, num_points_2d / BLOCK_SIZE);

  // Generates points
  generator<<<grid_dim, block_dim>>>(num_points_2d * num_points_2d, d_points_2d);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  // // Generates points inside a square
  // generate_square<<<grid_dim, block_dim>>>(d_points_2d, num_points_2d);
  // CUDA_CHK(cudaGetLastError());

  // CUDA_CHK(cudaDeviceSynchronize());

  // // Point3D *points_3d = (Point3D *)malloc(num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  // // CUDA_CHK(cudaMemcpy(points_3d, d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyDeviceToHost));

  // // Calculates sin of points
  // double *d_sin_result;
  // CUDA_CHK(cudaMalloc((void **)&d_sin_result, num_points_2d * num_points_2d * sizeof(double)));

  // calculate_sin<<<grid_dim, block_dim>>>(num_points_2d, d_points_2d, d_sin_result);
  // CUDA_CHK(cudaGetLastError());
  // CUDA_CHK(cudaDeviceSynchronize());

  double *gpu_special_sum_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_special_sum_result, num_points_2d * num_points_2d * sizeof(double)));

  special_sum<<<grid_dim, block_dim>>>(num_points_2d, gpu_special_sum_result, 1, d_points_2d);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *special_sum_result = (double *)malloc(num_points_2d * num_points_2d * sizeof(double));
  CUDA_CHK(cudaMemcpy(special_sum_result, gpu_special_sum_result, num_points_2d * num_points_2d * sizeof(double), cudaMemcpyDeviceToHost));

  // print_matrix_of_points(special_sum_result, 64);

  free(special_sum_result);
  // CUDA_CHK(cudaFree(d_sin_result));
  CUDA_CHK(cudaFree(d_points_2d));
  CUDA_CHK(cudaFree(gpu_special_sum_result));

  return 0;
}
