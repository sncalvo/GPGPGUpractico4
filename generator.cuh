#ifndef GENERATORS_H

#define GENERATORS_H

#include "./common.h"

__global__ void generate_square(Point2D *points, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    points[j + i * n] = {j * SMALL_POINT_SIZE - PI, i * SMALL_POINT_SIZE - PI};
  }
}

__global__ void generate_cube(Point3D *points, int n) {
  unsigned int i = blockIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int k = blockIdx.z;

  if (i < n && j < n && k < n) {
    points[i * n * n + j * n + k] = {i * BIG_POINT_SIZE - PI, j * BIG_POINT_SIZE - PI, k * BIG_POINT_SIZE - PI};
  }
}

#endif
