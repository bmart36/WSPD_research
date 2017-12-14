__host__ __device__ int msb(long int p)
{
	int high=31, low=0;

	while (high - low > 1)
	{
	    int mid = (high+low)/2;
	    int maskHigh = (1 << high) - (1 << mid);
	    if ((maskHigh & p) > 0)
	        low = mid;
	    else
	        high = mid;
	}
	return low;	       
}

int range(long int bc)
{
	int msb_bc = msb(bc);
	int bit = ((msb_bc+1) + 3 - 1) / 3;
  	return pow(2, bit);
}