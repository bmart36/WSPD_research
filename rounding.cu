class Node;
template<class type>
class Stack;

__global__ void rounding(float *d_f_points, int *d_points, Stack<Node> *d_roots)
{
	int index = threadIdx.x * 3;
	for(int i = 0; i<3; i++)
	{
		d_points[index + i]=floor(d_f_points[index + i]);
	}
}