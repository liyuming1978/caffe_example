#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(preprocess, Dtype)(const int_tp n,float scale,float mean,__global const float* x,
				     __global Dtype* out) {
	for (int_tp index = get_global_id(0); index < n; index += get_global_size(0))
	{
		out[out_index] = (x[in_index]+mean)*scale;
	}
}

