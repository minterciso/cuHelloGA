/*
 * common.h
 *
 *  Created on: 12/12/2018
 *      Author: minterciso
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <cuda.h>
#include <stdio.h>

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CHECK_RAND(x)\
{\
    	const curandStatus_t error = x;\
    	if(error != CURAND_STATUS_SUCCESS)\
    	{\
    		printf("Error: %s:%d", __FILE__, __LINE__);\
    		exit(1);\
    	}\
}

#define CHECK(call)															\
		{																	\
	const cudaError_t error=call;											\
	if(error != cudaSuccess){												\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
		}



#endif /* COMMON_H_ */
