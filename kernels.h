/*
 * kernels.h
 *
 *  Created on: 12/12/2018
 *      Author: minterciso
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <stdio.h>
#include <cuda.h>
#include "ga.h"

__global__ void fitness(char s_dest[LEN_SIZE], individual *pop);


#endif /* KERNELS_H_ */
