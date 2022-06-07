#include "./common.h"
#include "./generator.cuh"

/*
  special_sum calculates the sum of all values inside a matrix in a radius.

  num_points: number of points in a matrix
  sum_result: pointer to the result array
  radius: radius of elements to sum
  matrix: pointer to the matrix with values
*/
__global__ void special_sum(unsigned int num_points, double *sum_result, unsigned int radius, double *matrix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ double *matrix_point;

  if (i >= num_points || j >= num_points) {
    return;
  }

  matrix_point = &matrix[j + i * num_points];

  double result = -8 * (*matrix_point);

  for (int offset = -redius; offset <= radius; offset++) {
    matrix_point = &matrix[j + (i + offset) * num_points];
    result += *matrix_point;
  }

  for (int offset = -redius; offset <= radius; offset++) {
    matrix_point = &matrix[(j + offset2) + i * num_points];
    result += *matrix_point;
  }

  sum_result[j + i * num_points] = result / (SMALL_POINT_SIZE * SMALL_POINT_SIZE);
}

int main() {
  // Builds cube of points with space of BIG_POINT_SIZE
  unsigned int num_points_3d = trunc(SQUARE_LENGTH / BIG_POINT_SIZE);

  Point3D *d_points_3d;
  CUDA_CHK(cudaMalloc(&d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D)));

  // Generates points inside a cube
  dim3 block_dim_cube(1, 1, 1);
  dim3 grid_dim_cube(num_points_3d, num_points_3d, num_points_3d);
  generate_cube<<<grid_dim_cube, block_dim_cube>>>(d_points_3d, num_points_3d);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  // Point3D *points_3d = (Point3D *)malloc(num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  // CUDA_CHK(cudaMemcpy(points_3d, d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyDeviceToHost));


  // Calculates sin of points
  double *d_sin_result;
  CUDA_CHK(cudaMalloc((void **)&d_sin_result, num_points_2d * num_points_2d * sizeof(double)));

  calculate_sin<<<grid_dim, block_dim>>>(num_points_2d, d_points_2d, d_sin_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());


  double *gpu_special_sum_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_special_sum_result, num_points_2d * num_points_2d * sizeof(double)));

  special_sum<<<grid_dim, block_dim>>>(num_points_2d, gpu_special_sum_result, 1, d_sin_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *special_sum_result = (double *)malloc(num_points_2d * num_points_2d * sizeof(double));
  CUDA_CHK(cudaMemcpy(special_sum_result, gpu_special_sum_result, num_points_2d * num_points_2d * sizeof(double), cudaMemcpyDeviceToHost));

  print_matrix_of_points(special_sum_result, 64);

  return 0;
}
