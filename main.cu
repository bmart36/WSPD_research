#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <fstream>
#include <sstream>
#include "rounding.cu"
#include "shuffle.cu"
#include "build_tree.cu"
#include "wspd.cu"

using namespace std;

void readFile(const char *filename, float *points)
{
	string s, t, tmp;
	int n = 0;
	std::ifstream fin(filename);
	if (!fin.is_open()) cout << "Unable to open file!\n";

	while ( getline(fin, s) )
	{
		stringstream myline(s);
		myline >> t;
		if (t == "v")
		{
	        myline >> tmp;
			float x=stof(tmp.substr(0, tmp.find("/")));

			myline >> tmp;
			float y=stof(tmp.substr(0, tmp.find("/")));

			myline >> tmp;
			float z=stof(tmp.substr(0, tmp.find("/")));
			points[n] = x;
			points[n+1] = y;
			points[n+2] = z;
	        n+=3;
	 	}
	}
}

int main()
{
	unsigned int n = 12;

	// host variables
	float *h_f_points;
	long int *h_s_points;
	Node *h_root;
	Stack<Node> *h_quadtree = new Stack<Node>();
	Stack<Node2> *h_pairs = new Stack<Node2>();

	// device variables
	float *d_f_points;
	int *d_points;
	long int *d_s_points;
	Node *d_root;
	Stack<Node> *d_quadtree;
	Stack<Node> *d_roots;
	Stack<Node2> *d_queue;
	Stack<Node2> *wspd_pairs;

	// allocate host memory
	h_f_points = (float *)malloc(n * 3 * sizeof(float));
	h_s_points = (long int *)malloc(n * sizeof(long int));
	h_root = (Node *)malloc(sizeof(Node));

	// allocate device memory
	cudaMalloc((void **) &d_f_points, n * 3 * sizeof(float));
	cudaMalloc((void **) &d_points, n * 3 * sizeof(int));
	cudaMalloc((void **) &d_s_points, n * sizeof(long int));
	cudaMalloc((void **) &d_root, sizeof(Node));
	cudaMalloc((void **) &d_quadtree, sizeof(Stack<Node>));
	cudaMalloc((void **) &d_roots, sizeof(Stack<Node>));
	cudaMalloc((void **) &d_queue, sizeof(Stack<Node2>));
	cudaMalloc((void **) &wspd_pairs, sizeof(Stack<Node2>));

	// fill h_f_points array
	readFile("test.txt", h_f_points);

	// copy varibles to GPU
	cudaMemcpy(d_roots, h_quadtree, sizeof(Stack<Node>), cudaMemcpyHostToDevice);
	cudaMemcpy(d_quadtree, h_quadtree, sizeof(Stack<Node>), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f_points, h_f_points, n * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_queue, h_pairs, sizeof(Stack<Node2>), cudaMemcpyHostToDevice);
	cudaMemcpy(wspd_pairs, h_pairs, sizeof(Stack<Node2>), cudaMemcpyHostToDevice);

	// rounding points to a grid
	rounding<<<1, n>>>(d_f_points, d_points, d_roots);

	// shuffle points' coordinates
	shuffle<<<1, n>>>(d_points, d_s_points);

	// sort the points in invreasing order
	thrust::device_ptr<long int> t_d_s(d_s_points);
	if(!thrust::is_sorted(t_d_s, t_d_s + (n-1)))
	{
		thrust::stable_sort(t_d_s, t_d_s + n);
	}
 
	// copy ordered points to CPU
	cudaMemcpy(h_s_points, d_s_points, n * sizeof(long int), cudaMemcpyDeviceToHost);

	// size of the grid
	int w = range(h_s_points[n-1]);

	// computing compressed quadtrees
	compressed_quadtree<<< 1, BLOCK_SIZE>>>(d_s_points, n/BLOCK_SIZE, d_roots, d_quadtree, w, msb(h_s_points[n-1]));

	// merging of compressed quadtrees
	tree_merge<<<1 ,1>>>(d_roots, d_quadtree, d_root, w, msb(h_s_points[n-1]));

	// copy quadtree to CPU
	cudaMemcpy(h_quadtree, d_quadtree, sizeof(Stack<Node>), cudaMemcpyDeviceToHost);
	//cudaMemcpy(*h_root, d_root, sizeof(Node), cudaMemcpyDeviceToHost);


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

