/*
 * ga.c
 *
 *  Created on: 12/12/2018
 *      Author: minterciso
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#include "ga.h"
#include "kernels.h"
#include "common.h"

individual* create_population(curandGenerator_t gen)
{
	// cuRand should be already initialized
	int i,j;
	int min=VAL_MIN;
	int max=VAL_MAX;
	unsigned int *h_random, *d_random;
	size_t random_bytes = sizeof(unsigned int)*(LEN_SIZE*POP_SIZE);
	individual *pop = NULL;

	// Allocate random numbers array
	if((h_random=(unsigned int*)malloc(random_bytes))==NULL)
	{
		perror("Unable to allocate host memory for random numbers.");
		return NULL;
	}
	memset(h_random, '\0', random_bytes);
	CHECK(cudaMalloc((void**)&d_random, random_bytes));
	CHECK(cudaMemset(d_random,'\0', random_bytes));
	// Generate random numbers on device
	CHECK_RAND(curandGenerate(gen, d_random, LEN_SIZE*POP_SIZE));
	CHECK(cudaMemcpy(h_random, d_random, random_bytes, cudaMemcpyDeviceToHost));
	// Allocate memory for population
	if((pop=(individual*)malloc(sizeof(individual)*POP_SIZE))==NULL)
	{
		perror("Unable to allocate host memory for population");
		CHECK(cudaFree(d_random));
		free(h_random);
		return NULL;
	}
	memset(pop, '\0', sizeof(individual)*POP_SIZE);

	// Now finally create each string for each individual
	unsigned int rnd_idx = 0;
	for(i=0;i<POP_SIZE;i++)
	{
		for(j=0;j<LEN_SIZE;j++)
		{
			unsigned int rnd_val = (h_random[rnd_idx++] % (max-min+1)+min);
			pop[i].s[j] = (char)(rnd_val);
		}
	}
	// Free memory
	free(h_random);
	CHECK(cudaFree(d_random));
	return pop;
}

void destroy_population(individual *pop)
{
	if(pop != NULL)
		free(pop);
}

void xover_and_mutate(individual *pop, curandGenerator_t gen)
{
	int i,j;
	int min = VAL_MIN;
	int max = VAL_MAX;
	int qtdRandomInts = (POP_SIZE-KEEP_POP)/2 + (POP_SIZE-KEEP_POP);
	int qtdRandomMut = KEEP_POP*LEN_SIZE;
	int qtdRandomIntsPool = POP_SIZE*LEN_SIZE;
	unsigned int *h_randomInts, *d_randomInts;
	unsigned int *h_randomIntsPool, *d_randomIntsPool;
	unsigned int *h_randomMut, *d_randomMut;
	size_t s_randomInts = sizeof(unsigned int)*qtdRandomInts;
	size_t s_randomMut = sizeof(unsigned int)*qtdRandomMut;
	size_t s_randomIntsPool = sizeof(unsigned int)*qtdRandomIntsPool;

	// Kill low performance individuals
	for(i=KEEP_POP; i<POP_SIZE; i++)
	{
		pop[i].fitness = 0;
		memset(pop[i].s, '\0', LEN_SIZE);
	}

	// Allocate memory
	if((h_randomInts=(unsigned int*)malloc(s_randomInts))==NULL)
	{
		perror("Unable to allocate memory on host");
		return;
	}
	if((h_randomMut=(unsigned int*)malloc(s_randomMut))==NULL)
	{
		perror("Unable to allocate memory on host");
		free(h_randomInts);
		return;
	}
	if((h_randomIntsPool=(unsigned int*)malloc(s_randomIntsPool))==NULL)
	{
		perror("Unable to allocate memory on host");
		free(h_randomInts);
		free(h_randomMut);
		return;
	}
	memset(h_randomInts, '\0', s_randomInts);
	memset(h_randomMut, '\0', s_randomMut);
	memset(h_randomIntsPool, '\0', s_randomIntsPool);
	CHECK(cudaMalloc((void**)&d_randomInts, s_randomInts));
	CHECK(cudaMalloc((void**)&d_randomMut, s_randomMut));
	CHECK(cudaMalloc((void**)&d_randomIntsPool, s_randomIntsPool));
	// Create random numbers on device
	CHECK_RAND(curandGenerate(gen, d_randomInts, qtdRandomInts));
	CHECK_RAND(curandGenerate(gen, d_randomIntsPool, qtdRandomIntsPool));
	CHECK_RAND(curandGenerate(gen, d_randomMut, qtdRandomMut));
	CHECK(cudaMemcpy(h_randomInts, d_randomInts, s_randomInts, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_randomMut, d_randomMut, s_randomMut, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_randomIntsPool, d_randomIntsPool, s_randomIntsPool, cudaMemcpyDeviceToHost));
	// Now xover and mutate, based on the random numbers generated
	int rndIntsIdx = 0;
	int rndFloatsIdx = 0;
	int rndIntsPoolIdx = 0;
	float rndMut = 0.0;
	for(i=KEEP_POP; i<POP_SIZE; i+=2)
	{
		unsigned int id1=h_randomInts[rndIntsIdx++] % KEEP_POP;
		unsigned int id2=h_randomInts[rndIntsIdx++] % KEEP_POP;
		individual *p1 = &pop[id1], *p2 = &pop[id2];
		individual *s1 = &pop[i], *s2 = &pop[i+1];
		unsigned int xp = h_randomInts[rndIntsIdx++]%LEN_SIZE;
		memcpy(s1->s, p1->s, xp);
		memcpy(s1->s + xp, p2->s + xp, (LEN_SIZE-xp));
		memcpy(s2->s, p2->s, xp);
		memcpy(s2->s + xp, p1->s +xp, (LEN_SIZE-xp));
		// Mutate
		for(j=0;j<LEN_SIZE;j++)
		{
			rndMut = (float)h_randomMut[rndFloatsIdx++]/(float)(RAND_MAX);
			if(rndMut < PROB_MUT)
				s1->s[j] = (char)(h_randomIntsPool[rndIntsPoolIdx++] % (max-min+1)+min);
			rndMut = (float)h_randomMut[rndFloatsIdx++]/(float)(RAND_MAX);
			if(rndMut < PROB_MUT)
				s2->s[j] = (char)(h_randomIntsPool[rndIntsPoolIdx++] % (max-min+1)+min);

		}
	}
	// Before stoping, zero the fitness of the best ones
	for(i=0;i<KEEP_POP;i++)
	{
		pop[i].fitness=0;
	}
	free(h_randomInts);
	free(h_randomMut);
	free(h_randomIntsPool);
	CHECK(cudaFree(d_randomMut));
	CHECK(cudaFree(d_randomInts));
	CHECK(cudaFree(d_randomIntsPool));
}
