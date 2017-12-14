class Bounding_box
{
	public:
		int min_x;
		int min_y;
		int min_z;
		int max_x;
		int max_y;
		int max_z;

		__host__ __device__ Bounding_box(int w, int level, int p1, int p2, int msb)
		{
            min_x = 0;
		    min_y = 0;
		    min_z = 0;
		    max_x = w;
		    max_y = w;
		    max_z = w;
            int x;
            
            for(int i=msb; i>level; i--)
            {
            	//z
            	if(i%3 == 0)
            	{
            		x = (max_z - min_z)/2;
            		if(p1 & (1 << i))
            			min_z  += x;
            		else
            			max_z -= x;
            	}
            	else if((i+2)%3 == 0)
            	{
            		x = (max_y - min_y)/2;
            		if(p1 & (1 << i))
            			min_y += x;
            		else
            			max_y -= x;
            	}
            	//x
            	else
            	{
            		x = (max_x - min_x)/2;
            		if((p1 & (1 << i)))
            			min_x += x;
            		else
            			max_x -= x;
            	}
            }
            max_x--;
            max_y--;
            max_z--;
        }

        __host__ __device__ float3 getCenter()
        {
        	float3 c;
        	c.x = (max_x - min_x)/2.0;
        	c.y = (max_y - min_y)/2.0;
        	c.z = (max_z - min_z)/2.0;
        	c.x += min_x;
        	c.y += min_y;
        	c.z += min_z;

        	return c;
        }

        __host__ __device__ void printBox()
        {
            printf("x: %d, %d\n", min_x, max_x );
            printf("y: %d, %d\n", min_y, max_y );
            printf("z: %d, %d\n", min_z, max_z );
        }

};
