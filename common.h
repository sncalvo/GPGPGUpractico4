#ifndef COMMON_H

#define COMMON_H

#include <math.h>

#include <locale.h>

#include "cuda.h"

#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 16
#define PI 3.14159265358

#define SQUARE_LENGTH 2 * PI
#define SMALL_POINT_SIZE 0.001
// #define BIG_POINT_SIZE 0.1
#define BIG_POINT_SIZE 0.01

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Point2D {
  double x;
  double y;
};

struct Point3D {
  double x;
  double y;
  double z;
};

void print_matrix_of_points(Point2D *points, int n) {
  printf("\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, %f ", points[i*n + j].x, points[i * n + j].y);
    }
    printf("\n");
  }

  printf("\n");
}

void print_matrix_of_points(Point3D *points, int n) {
  printf("\n");

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        printf("%f, %f, %f ", points[i * n * n + j * n + k].x, points[i * n * n + j * n + k].y, points[i * n * n + j * n + k].z);
      }
      printf("\n");
    }
  }

  printf("\n");
}

void print_matrix_of_points(double *points, int n) {
  printf("\n");

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", points[i * n + j]);
    }
    printf("\n");
  }
}

void print_matrix_of_points_with_origin(Point2D *origins, double *points, int n) {
  // printf("\n");

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, %f\n", origins[i * n + j].x + origins[i * n + j].y, points[i * n + j]);
    }
    // printf("\n");
  }
}

void print_matrix_of_points3D(double *points, int n) {
  printf("\n");

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        printf("%f ", points[i * n * n + j * n + k]);
      }
      printf("\n");
    }
  }

  printf("\n");
}

#endif
