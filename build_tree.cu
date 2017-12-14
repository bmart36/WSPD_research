#include "Node.cu"
#include "stack.h"

__host__ __device__ void treePrint(Node *n)
{
	if(n != NULL)
	{
		treePrint(n->left);
		n->printNode();
		treePrint(n->right);
	}
}

//function to find leftmost leaf of a tree with root n
__host__ __device__ Node* leftmost(Node *n)
{
	if(n->left != NULL)
		return leftmost(n->left);
	else
		return n;
}

//function to find rightmost leaf of a tree with root n
__host__ __device__ Node* rightmost(Node *n)
{
	if(n->right != NULL)
		return rightmost(n->right);
	else
		return n;
}

// compressed quadtree is computed for a given set of points
__global__ void compressed_quadtree(long int *d_points, int n, Stack<Node> *d_roots, Stack<Node> *d_quadtree,  int w, int msb)
{
	int index = threadIdx.x * n;
	// initialize root of the tree to biggest possible box in level w and child p1
	Node *r = new Node;
	Node *p = new Node(r, d_points[index]);
	d_quadtree->push(p);

	r->right = p;
	r->level = w;

	Node *temp = new Node;
	temp = r;

	for(int i=1; i<n; i++)
	{
		// new nodes created in each iteration q(pi-1, pi) and pi
		Node *q = new Node;
		Node *p_i = new Node(q, d_points[i+index]);
		d_quadtree->push(q);
		d_quadtree->push(p_i);

		// set the level to most significant bit different between 2 points
		q->setLevel(d_points[i+index-1], d_points[i+index]);
		q->data = d_points[i+index-1];
		q->box = new Bounding_box(w, q->level, d_points[i+index-1], d_points[i+index], msb);
		q->right = p_i;

		// find location for new node
		while (q->level > r->level)
		  r = r->parent;

		q->left = r->right;
		r->right->parent = q;
		q->parent = r;
		r->right = q;

		// preserve newest q node to compare in the future iteration
		r = q;
	}

	// root of the tree is stored in stack
	temp = temp->right;
	temp->parent = NULL;
	d_roots->insert(temp, threadIdx.x);
}

// compressed trees are merged into a sole compressed quadtree
__global__ void tree_merge(Stack<Node> *d_roots, Stack<Node> *d_quadtree, Node *d_root, int w, int msb)
{
	while(d_roots->get_size()>0)
	{
		// take previously built trees to merge
		Node *t1 = d_roots->pop();
		Node *t2 = d_roots->pop();

		if(!t1 || !t2)
			return;


		// if t2 contains smaller points than t1 swap
		if(t1->data > t2->data)
		{
			Node *temp = t1;
			t1 = t2;
			t2 = temp;
		}

		// get rightmost leaf of t1 and leftmost leaf of t2
		Node *r = rightmost(t1);
		Node *l = leftmost(t2);

		// righmost and leftmost leaves parents
		Node *r_p = new Node;
		Node *l_p = new Node;

		// new node which will merge t1 and t2
		Node *m = new Node;
		d_quadtree->push(m);
		m->setLevel(r->data, l->data);
		m->data = r->data;
		m->box = new Bounding_box(w, m->level, r->data, l->data, msb);

		// find new node's left child in t1
		while(r->parent && m->level > r->parent->level)
			r = r->parent;
		m->left = r;
		r_p = r->parent;
		r->parent = m;

		// find new node's right child in t2
		while(l->parent && m->level > l->parent->level)
			l = l->parent;
		m->right = l;
		l_p = l->parent;
		l->parent = m;

		// test to find new node's parent
		if(!l_p && !r_p)
		{			
			d_roots->push(m);
		}
		else if(!r_p && l_p)
		{

			m->parent = l_p;
			l_p->left = m;
			d_roots->push(l_p);
		}
		else if(!l_p && r_p)
		{
			m->parent = r_p;
			r_p->right = m;
			d_roots->push(r_p);
		}
		else if(r_p->level > l_p->level)
		{
			m->parent = l_p;
			l_p->left = m;
			while(l_p->parent && r_p->level > l_p->level)
			{
				l_p = l_p->parent;
			}
			l_p->parent = r_p;
			r_p->right = l_p;
			d_roots->push(r_p);
		}
		else if(l_p->level > r_p->level)
		{
			m->parent = r_p;
			r_p->right = m;
			while(r_p->parent && l_p->level > r_p->level)
			{
				r_p = r_p->parent;
			}
			r_p->parent = l_p;
			l_p->left = r_p;
			d_roots->push(l_p);
		}
	}
	// root of the compressed quadtree is stored in variable
	d_root = d_roots->top();
	treePrint(d_root);
}
