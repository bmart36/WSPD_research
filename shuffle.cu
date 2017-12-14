#include "3D_LUT.h"

__global__ void shuffle(int *d_points, long int *d_s_points)
{
	int index = threadIdx.x * 3;

	d_s_points[threadIdx.x] = morton_x[d_points[index]] | morton_y[d_points[index + 1]] | morton_z[d_points[index + 2]];
}