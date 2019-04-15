//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void minValue(__global int * myArray, __global int * mins, __local int* scratch) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = myArray[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid + i] < scratch[lid])
				scratch[lid] = scratch[lid + i];
				
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//copy the cache to output array

	if (!lid) {
		atomic_min(&mins[0], scratch[lid]);
	}
}

//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void maxValue(__global int * myArray, __global int * mins, __local int* scratch) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = myArray[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid + i] > scratch[lid])
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//copy the cache to output array

	if (!lid) {
		atomic_max(&mins[0], scratch[lid]);
	}
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
kernel void sum(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void variation(global const int* A, global const int* B, global int* C, int maxValue) {
	int id = get_global_id(0);

	if (id < maxValue) {
		C[id] = A[id] - B[id];
		C[id] = C[id] * C[id];
	}
}

kernel void parallelSelection(__global const int *A, __global int *B)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	float ikey = A[id];

	// loop through all other data points to find new position of current element
	int pos = 0;
	for	(int j = 0; j < N; j++)
	{
		float jkey = A[j];
		bool smaller = (jkey < ikey) || (jkey == ikey && j < id);
		// calculate new position
		pos += (smaller)?1:0;
	}
	// place in the new position
	B[pos] = ikey;
}

__kernel void ParallelSelection_Blocks(__global const int * A, __global int * B, __local int * scratch)
{
	int i = get_global_id(0); // current thread
	int n = get_global_size(0); // input size
	int wg = get_local_size(0); // workgroup size

	float ikey = A[i];

	int blockSize = 1024 * wg; // block size

	// Compute position of iKey in output
	int pos = 0;
	// Loop on blocks of size BLOCKSIZE keys (BLOCKSIZE must divide N)
	for (int j = 0; j < n; j += blockSize)
	{
		// Load BLOCKSIZE keys using all threads (BLOCK_FACTOR values per thread)
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int index = get_local_id(0); index < blockSize; index += wg)
			scratch[index] = A[j + index];
		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop on all values in AUX
		for (int index = 0; index < blockSize; index++)
		{
			uint jKey = scratch[index]; // broadcasted, local memory
			bool smaller = (jKey < ikey) || (jKey == ikey && (j + index) < i); // in[j] < in[i] ?
			pos += (smaller) ? 1 : 0;
		}
	}
	B[pos] = ikey;
}