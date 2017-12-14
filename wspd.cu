#define max(a, b) ((a) > (b))? (a): (b)
class Node;

//distance from point to point
__host__ __device__ float distance(float3 c1, float3 c2)
{
	float x, y, z;
	x = pow(c1.x - c2.x, 2);
	y = pow(c1.y - c2.y, 2);
	z = pow(c1.z - c2.z, 2);
	
	return sqrt(x + y + z);
}

__host__ __device__ float radius(Node *q)
{
	// if the node is a leaf the diameter returned is 0
	if(q->level == -1)
		return 0;

	// the radius is the distance between the box center and one of the box's corner
	float3 box_center;
	box_center.x = q->box->min_x;
	box_center.y = q->box->min_y;
	box_center.z = q->box->min_z;
	return distance(q->nodeCenter(), box_center);
}

// all pairs with same parent are put in the WSPD queue
__global__ void pre_wspd(Stack<Node> *d_quadtree, Stack<Node2> *d_queue)
{
	int n = ((d_quadtree->get_size()+1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int index = threadIdx.x * n;
	Node *temp;

	for(int i=index; i<index+n; i++)
	{
		temp = d_quadtree->get(i);
		if(!temp)
			return;

		// if node is leaf
		if(temp->level == -1)
			continue;

		Node2 *n = new Node2;

		n->n1 = temp->left;
		n->n2 = temp->right;
		d_queue->push(n);
	}
	printf("queue size: %d\n", d_queue->get_size()+1);
}

// WSPD pairs are computed from the pairs existing in the queue
__global__ void wspd(float d_e, Stack<Node2> *d_queue, Stack<Node2> *wspd_pairs)
{
	int i=1;
	float radius_q1, radius_q2, dist, max;
	while(d_queue->get_size()>=0)
	{
		printf("Iteration: %d\n", i);

		Node2 *pair = d_queue->pop();
		Node *q1 = pair->n1;
		//q1->printNode();
		Node *q2 = pair->n2;
		//q2->printNode();

		Node2 *q1_q2_left = new Node2;
		Node2 *q1_q2_right = new Node2;

		radius_q1 = radius(q1);
		printf("radius 1: %f\n", radius_q1);
		radius_q2 = radius(q2);
		printf("radius 2: %f\n", radius_q2);
		dist = distance(q1->nodeCenter(), q2->nodeCenter()) - radius_q1 - radius_q2;
		printf("Distance: %f\n", dist);
		max = max(2*radius_q1, 2*radius_q2);
		printf("max: %f\n", (d_e * max));

		if(d_e * dist >= max)
		{
			printf("I'm here 1\n\n");
			wspd_pairs->push(pair);
		}
		else if(q1->level >= q2->level)
		{
			printf("I'm here 2\n\n");
			q1_q2_left->n1 = q1->left;
			q1_q2_left->n2 = q2;
			d_queue->push(q1_q2_left);
			q1_q2_right->n1 = q1->right;
			q1_q2_right->n2 = q2;
			d_queue->push(q1_q2_right);
		}
		else
		{
			printf("I'm here 3\n\n");
			q1_q2_left->n1 = q1;
			q1_q2_left->n2 = q2->left;
			d_queue->push(q1_q2_left);
			q1_q2_right->n1 = q1;
			q1_q2_right->n2 = q2->right;
			d_queue->push(q1_q2_right);
		}
		i++;
	}
	printf("%d\n", wspd_pairs->get_size());
}