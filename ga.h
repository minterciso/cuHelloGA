/*
 * ga.h
 *
 *  Created on: 12/12/2018
 *      Author: minterciso
 */

#ifndef GA_H_
#define GA_H_

#include <stdio.h>
#include <curand.h>
#include "consts.h"

typedef struct
{
	char s[LEN_SIZE];
	unsigned int fitness;
} individual;

individual* create_population(curandGenerator_t gen);
void destroy_population(individual *pop);
void xover_and_mutate(individual *pop, curandGenerator_t gen);


#endif /* GA_H_ */
