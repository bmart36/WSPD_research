__constant__ const int BLOCK_SIZE = 3;
__constant__ const int STACK_MAX = 1000000000;

#include "box.h"
#include "functions.h"

__host__ __device__ float morton_1(long int x) {

    x = x & 0x9249249249249249;
    x = (x | (x >> 2))  & 0x30c30c30c30c30c3;
    x = (x | (x >> 4))  & 0xf00f00f00f00f00f;
    x = (x | (x >> 8))  & 0x00ff0000ff0000ff;
    x = (x | (x >> 16)) & 0xffff00000000ffff;
    return (float) x;
}

__host__ __device__ float3 d_morton(long int d)
{
	float3 xyz;
    xyz.x = morton_1(d >> 2);
    xyz.y = morton_1(d >> 1);
    xyz.z = morton_1(d);

    return xyz;
}

class Node
{
	public:
		Node *left;
		Node *right;
		Node *parent;
		long int data;
		int level;
		Bounding_box *box;

		__host__ __device__ Node() : left(NULL), right(NULL), parent(NULL), data(0), level(-1), box(NULL) {}
		__host__ __device__ Node(Node *p, long int point) : parent(p), data(point), left(NULL), right(NULL), level(-1), box(NULL) {}

		__host__ __device__ void setLevel(long int p1, long int p2)
		{
			long int temp = p1 xor p2;

			level = msb(temp);
		}

		__host__ __device__ float3 nodeCenter()
		{
			if(box)
				return box->getCenter();
			else
				return d_morton(data);
		}

		__host__ __device__ void printNode()
		{
			printf("Node: %p\n", this);
			printf("Left: %p\n", left);
			printf("Right: %p\n", right);
			printf("Parent: %p\n", parent);
			printf("Data: %ld\n", data);
			printf("Level: %d\n", level);
			if(box) box->printBox();
			float3 c = nodeCenter();
			printf("center: %f, %f, %f\n", c.x, c.y, c.z);
			printf("\n");
		}
};

struct Node2
{
	Node *n1;
	Node *n2;
};
