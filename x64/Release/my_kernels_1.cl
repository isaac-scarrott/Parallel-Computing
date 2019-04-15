//Calculates the minimmum integer value using local memory - faster than global
kernel void minValue(__global int * tempuratures, __global int * mins, __local int* scratch) {

	int id = get_global_id(0); //The global work-item ID specifies the work-item ID based on the number of global work-items specified to execute the kernel
	int lid = get_local_id(0); //Returns the unique local work-item ID
	int N = get_local_size(0); //Returns the number of local work-items

	scratch[lid] = tempuratures[id]; //Copies it from global to local memort

	barrier(CLK_LOCAL_MEM_FENCE); //Sync

	//Goes through in strides that get bigger and bigger checking if the item next is smaller 
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid + i] < scratch[lid])
				scratch[lid] = scratch[lid + i];
				
		barrier(CLK_LOCAL_MEM_FENCE); //sync
	}

	//Once all of the workgroups have been searched for min item this will find the min of all the values
	if (!lid) {
		atomic_min(&mins[0], scratch[lid]);
	}
}

//Calculates the minimmum float value using local memory - faster than globa
kernel void minValueFloat(__global float * tempuratures, __global float * mins, __local float* scratch) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0); //Returns the unique global work-item ID value
	int ng = get_num_groups(0); //Returns the number of work-groups that will execute a kernel

	scratch[lid] = tempuratures[id];

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid + i] < scratch[lid])
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	//Once all of the workgroups have been searched for min item this will find the min of all the values. It does this by checking the first value of each workgroup and seeing which is smallest
	if (!lid) {
		mins[gid] = scratch[lid];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int i = 1; i < ng; ++i) {
			if (mins[0] > mins[i]) {
				mins[0] = mins[i];
			}
		}
	}
}

//Calculates the maxmimum integer value using local memory - faster than global. Same as min but for max
kernel void maxValue(__global int * tempuratures, __global int * maxs, __local int* scratch) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = tempuratures[id];

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid + i] > scratch[lid])
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_max(&maxs[0], scratch[lid]);
	}

}

//Calculates the maxmimum float value using local memory - faster than global. Same as min but for max
kernel void maxValueFloat(__global float * tempuratures, __global float * maxs, __local float* scratch) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0);
	int ng = get_num_groups(0);

	scratch[lid] = tempuratures[id];

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid + i] > scratch[lid])
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		maxs[gid] = scratch[lid];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int i = 1; i < ng; ++i) {
			if (maxs[0] < maxs[i]) {
				maxs[0] = maxs[i];
			}
		}
	}

}

//Sums up the tempuratures integers vector using local memory and outputs it in the 0 index in the output vector. Same as min but adds values up
kernel void sum(global const int* tempuratures, global int* output, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = tempuratures[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_add(&output[0], scratch[lid]);
	}
}

//Sums up the tempuratures floats vector using local memory and outputs it in the 0 index in the output vector. Same as min but adds values up
kernel void sumFloat(global const float* tempuratures, global float* output, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0);
	int ng = get_num_groups(0);

	scratch[lid] = tempuratures[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}


	if (!lid) {
		output[gid] = scratch[lid];
		barrier(CLK_LOCAL_MEM_FENCE);
		if (id == 0)
			for (int i = 1; i < ng; ++i) {
				output[0] += output[i];
			}
		}
	}

//This is acutally sum of squared error for each value in the tempuratures integer vector. This is done on the global memory
kernel void variation(global const int* tempurature, global const int* mean, global int* output, int maxValue) {
	int id = get_global_id(0);
	//If it's below the padding
	if (id < maxValue) {
		output[id] = tempurature[id] - mean[id];
		output[id] = output[id] * output[id];
	}
}

//This is acutally sum of squared error for each value in the tempuratures float vector. This is done on the global memory
kernel void variationFloat(global const float* tempurature, global const float* mean, global float* output, int maxValue) {
	int id = get_global_id(0);

	if (id < maxValue) {
		output[id] = tempurature[id] - mean[id];
		output[id] = output[id] * output[id];
	}
}

//Parallel selection sort on integers using the global memory
//http://www.bealto.com/gpu-sorting_parallel-selection.html
kernel void parallelSelection(__global const int *tempuraturesUnsorted, __global int *tempuraturesSorted)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	int ikey = tempuraturesUnsorted[id];

	int pos = 0;
	for	(int j = 0; j < N; j++)
	{
		float jkey = tempuraturesUnsorted[j];
		bool smaller = (jkey < ikey) || (jkey == ikey && j < id);
		pos += (smaller)?1:0;
	}

	tempuraturesSorted[pos] = ikey;
}

//Parallel selection sort on floats using the global memory
//http://www.bealto.com/gpu-sorting_parallel-selection.html
kernel void parallelSelectionFloat(__global const float *tempuraturesUnsorted, __global float *tempuraturesSorted)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	float ikey = tempuraturesUnsorted[id];

	int pos = 0;
	for (int j = 0; j < N; j++)
	{
		float jkey = tempuraturesUnsorted[j];
		bool smaller = (jkey < ikey) || (jkey == ikey && j < id);
		pos += (smaller) ? 1 : 0;
	}
	tempuraturesSorted[pos] = ikey;
}

//Parallel selection sort on integers using the local memory - faster
//http://www.bealto.com/gpu-sorting_parallel-selection.html
kernel void parallelSelectionLocal(__global const int *tempuraturesUnsorted, __global int * tempuraturesSorted, __local int * scratch)
{
	int i = get_global_id(0);
	int n = get_global_size(0);
	int wg = get_local_size(0);

	int ikey = tempuraturesUnsorted[i];


	int pos = 0;
	for (int j = 0; j < n; j += wg)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int index = get_local_id(0); index < wg; index += wg)
			scratch[index] = tempuraturesUnsorted[j + index];
		barrier(CLK_LOCAL_MEM_FENCE);


		for (int index = 0; index < wg; index++)
		{
			int jKey = scratch[index];
			bool smaller = (jKey < ikey) || (jKey == ikey && (j + index) < i);
			pos += (smaller) ? 1 : 0;
		}
	}
	tempuraturesSorted[pos] = ikey;
}

//Parallel selection sort on floats using the local memory - faster
//http://www.bealto.com/gpu-sorting_parallel-selection.html
kernel void parallelSelectionLocalFloat(__global const float *tempuraturesUnsorted, __global float * tempuraturesSorted, __local float * scratch)
{
	int i = get_global_id(0);
	int n = get_global_size(0);
	int wg = get_local_size(0);

	float ikey = tempuraturesUnsorted[i];


	int pos = 0;
	for (int j = 0; j < n; j += wg)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int index = get_local_id(0); index < wg; index += wg)
			scratch[index] = tempuraturesUnsorted[j + index];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int index = 0; index < wg; index++)
		{
			int jKey = scratch[index]; 
			bool smaller = (jKey < ikey) || (jKey == ikey && (j + index) < i); 
			pos += (smaller) ? 1 : 0;
		}
	}
	tempuraturesSorted[pos] = ikey;
}