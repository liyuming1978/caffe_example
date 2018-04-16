#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(preprocess, Dtype)(const int_tp n,float scale,float mean,__global const float* indata,
				     __global Dtype* outdata) {
	for (int_tp index = get_global_id(0); index < n; index += get_global_size(0))
	{
		outdata[index] = (indata[index]+mean)*scale;
	}
}

