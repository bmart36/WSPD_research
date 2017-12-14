#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "build_tree.cu"

__device__ int morton_x[16] =
{
    0, 2, 8, 10, 32, 34, 40, 42, 128, 130, 136, 138, 160, 162, 168, 170
};

__device__ int morton_y[16] =
{
    0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85
};

__global__ void rounding(float *d_f_points, int *d_points, int d, Stack<Node> *d_roots)
{
	int index = threadIdx.x * d;
	for(int i = 0; i<d; i++)
	{
		d_points[index + i]=floor(d_f_points[index + i]);
	}
}

__global__ void shuffle(int *d_points, int *d_s_points, int d)
{
	int index = threadIdx.x * d;

	d_s_points[threadIdx.x] = morton_x[d_points[index]] | morton_y[d_points[index + 1]];
}

int range(int bc)
{
	int msb_bc = msb(bc);
	int bit = (msb_bc+1)/2;
  	return pow(2, bit);
}

int main()
{
	unsigned int n = 12;
	unsigned int d = 2;

	// host variables
	float *h_f_points;
	int *h_s_points;
	Node *h_root;
	Stack<Node> *h_quadtree = new Stack<Node>();
	Stack<Node2> *h_pairs = new Stack<Node2>();

	// device variables
	float *d_f_points;
	int *d_points;
	int *d_s_points;
	Node *d_root;
	Stack<Node> *d_quadtree;
	Stack<Node> *d_roots;
	Stack<Node2> *d_queue;
	Stack<Node2> *wspd_pairs;

	// allocate host memory
	h_f_points = (float *)malloc(n * d * sizeof(float));
	h_s_points = (int *)malloc(n * sizeof(int));
	h_root = (Node *)malloc(sizeof(Node));

	// allocate device memory
	cudaMalloc((void **) &d_f_points, n * d * sizeof(float));
	cudaMalloc((void **) &d_points, n * d * sizeof(int));
	cudaMalloc((void **) &d_s_points, n * sizeof(int));
	cudaMalloc((void **) &d_root, sizeof(Node));
	cudaMalloc((void **) &d_quadtree, sizeof(Stack<Node>));
	cudaMalloc((void **) &d_roots, sizeof(Stack<Node>));
	cudaMalloc((void **) &d_queue, sizeof(Stack<Node2>));
	cudaMalloc((void **) &wspd_pairs, sizeof(Stack<Node2>));

	// fill h_f_points array
	h_f_points[0] = 1.2;
	h_f_points[1] = 5.2;
	h_f_points[2] = 3.4;
	h_f_points[3] = 12.4;
	h_f_points[4] = 10.10;
	h_f_points[5] = 14.10;
	h_f_points[6] = 10.5;
	h_f_points[7] = 1.5;
	h_f_points[8] = 12.11;
	h_f_points[9] = 12.11;
	h_f_points[10] = 13.6;
	h_f_points[11] = 4.6;	
	h_f_points[12] = 5.3;
	h_f_points[13] = 5.3;
	h_f_points[14] = 10.7;
	h_f_points[15] = 10.7;
	h_f_points[16] = 2.1;
	h_f_points[17] = 1.1;
	h_f_points[18] = 10.8;
	h_f_points[19] = 11.8;
	h_f_points[20] = 15.12;
	h_f_points[21] = 15.12;
	h_f_points[22] = 11.9;
	h_f_points[23] = 11.9;

	// copy varibles to GPU
	cudaMemcpy(d_roots, h_quadtree, sizeof(Stack<Node>), cudaMemcpyHostToDevice);
	cudaMemcpy(d_quadtree, h_quadtree, sizeof(Stack<Node>), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f_points, h_f_points, n * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_queue, h_pairs, sizeof(Stack<Node2>), cudaMemcpyHostToDevice);
	cudaMemcpy(wspd_pairs, h_pairs, sizeof(Stack<Node2>), cudaMemcpyHostToDevice);

	// rounding points to a grid
	rounding<<<1, n>>>(d_f_points, d_points, d, d_roots);

	// shuffle points' coordinates
	shuffle<<<1, n>>>(d_points, d_s_points, d);

	// sort the points in invreasing order
	thrust::device_ptr<int> t_d_s(d_s_points);
	if(!thrust::is_sorted(t_d_s, t_d_s + (n-1)))
	{
		thrust::stable_sort(t_d_s, t_d_s + n);
	}
 
	// copy ordered points to CPU
	cudaMemcpy(h_s_points, d_s_points, n * sizeof(int), cudaMemcpyDeviceToHost);

	// size of the grid
	int w = range(h_s_points[n-1]);

	// computing compressed quadtrees
	compressed_quadtree<<< 1, BLOCK_SIZE>>>(d_s_points, n/BLOCK_SIZE, d_roots, d_quadtree, w);

	// merging of compressed quadtrees
	tree_merge<<<1 ,1>>>(d_roots, d_quadtree, d_root, w);

	// copy quadtree to CPU
	cudaMemcpy(h_quadtree, d_quadtree, sizeof(Stack<Node>), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_root, d_root, sizeof(Node *), cudaMemcpyDeviceToHost);

	//treePrint(h_root);
	//printf("%d\n", h_quadtree->get_size());
	//h_root->printNode();
	
	//for(int i =0; i<=h_quadtree->get_size();i++)
	//	h_quadtree->get(i)->printNode();

	// computing WSPD
	pre_wspd<<<1, BLOCK_SIZE>>>(d_quadtree, d_queue);
	wspd<<<1, 1>>>(1, d_queue, wspd_pairs);

	cudaFree(d_f_points);
	cudaFree(d_points);
	cudaFree(d_s_points);
	cudaFree(d_roots);
	cudaFree(d_quadtree);
	cudaFree(d_queue);
	cudaFree(wspd_pairs);

	// return 0;
}

