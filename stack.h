template<class type>
class Stack
{
		type *items[STACK_MAX];
		int size;

  	public:
	  	__host__ __device__ Stack(): size(-1) {}

	    __device__ void push(type *item)
	  	{
	  		if(size+1>STACK_MAX)
	  			return;
	  		items[atomicAdd(&size, 1)+1] = item;
	  	}

	  	__device__ void insert(type *item, int index)
	  	{
	  		items[index] = item;
	  		atomicAdd(&size, 1);
	  	}

	  	__device__ type* pop()
	  	{
	  		if(size<0)
	  			return NULL;
	  		return items[atomicSub(&size, 1)];
	  	}

	  	__host__ __device__ type* get(int index)
	  	{
	  		return items[index];
	  	}

	  	__host__ __device__ int get_size()
	  	{
	  		return size;
	  	}

	  	__host__ __device__ type* top()
	  	{
	  		return items[size];
	  	}
};
