/*
 * kernels.cu
 *
 *  Created on: 12/12/2018
 *      Author: minterciso
 */

#include "kernels.h"
#include "ga.h"

__global__ void fitness(char s_dest[LEN_SIZE], individual *pop)
{
	unsigned int pop_idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int str_idx = threadIdx.y + blockDim.y * blockIdx.y;
	individual *ind = NULL;
	if(pop_idx < POP_SIZE && str_idx < LEN_SIZE)
	{
		ind = &pop[pop_idx];
		unsigned int l_fit = abs( (int)ind->s[str_idx] - (int)s_dest[str_idx]);
		atomicAdd(&ind->fitness, l_fit);
	}
}
