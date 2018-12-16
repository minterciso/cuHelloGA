/*
 * main.c
 *
 *  Created on: 09/12/2018
 *      Author: minterciso
 */

#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "ga.h"
#include "kernels.h"
#include "common.h"
#include "consts.h"
#include "utils.h"

static int cmpind(const void *p1, const void *p2)
{
	individual *i1 = (individual*)p1;
	individual *i2 = (individual*)p2;
	return i1->fitness - i2->fitness;
}

int main(int argc, char **argv)
{
	int dev = 0;
	CHECK(cudaSetDevice(dev));
	curandGenerator_t gen;
	individual *h_pop = NULL, *d_pop = NULL;
	char h_dest[LEN_SIZE], *d_dest;
	int g;
	double timeStart = 0.0, progStart = 0.0;

	progStart = cpuSecond();
	fprintf(stdout, "[*] Starting %s...\n",argv[0]);
	fprintf(stdout, "[*] Setting up Random generators...");
	timeStart = cpuSecond();
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
	fprintf(stdout, "[OK] (%0.5f)\n", cpuSecond()-timeStart);

	fprintf(stdout, "[*] Creating population...");
	timeStart = cpuSecond();
	if((h_pop=create_population(gen))==NULL)
	{
		fprintf(stderr,"Unable to create population on host!");
		return EXIT_FAILURE;
	}
	fprintf(stdout, "[OK] (%0.5f)\n", cpuSecond() - timeStart);
	CHECK(cudaMalloc((void**)&d_dest, sizeof(char)*LEN_SIZE));
	CHECK(cudaMalloc((void**)&d_pop, sizeof(individual)*POP_SIZE));
	strncpy(h_dest, "Hello World from a motherf****ing perspective!", LEN_SIZE);
	CHECK(cudaMemcpy(d_dest, h_dest, sizeof(char)*LEN_SIZE, cudaMemcpyHostToDevice));

	individual *best = NULL;
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(POP_SIZE/threadsPerBlock.x+1, LEN_SIZE/threadsPerBlock.y+1);
	fprintf(stdout,"[*] Using Kernel parameters: <<<(%d,%d),(%d,%d)>>>\n",numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
	fprintf(stdout,"[*] Evolving...");
	timeStart = cpuSecond();
	for(g=0;g<MAX_GEN;g++)
	{
		CHECK(cudaMemcpy(d_pop, h_pop, sizeof(individual)*POP_SIZE, cudaMemcpyHostToDevice));
		fitness<<<numBlocks, threadsPerBlock>>>(d_dest, d_pop);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaMemcpy(h_pop, d_pop, sizeof(individual)*POP_SIZE, cudaMemcpyDeviceToHost));
		qsort(h_pop, POP_SIZE, sizeof(individual), cmpind);
		best = &h_pop[0];
		/*
		if(best->fitness == 0)
			break;
			*/
		if(g == MAX_GEN)
			break;
		xover_and_mutate(h_pop, gen);
	}
	fprintf(stdout,"[OK] (%0.5f)\n", cpuSecond() - timeStart);
	if(best->fitness!=0)
		fprintf(stdout, "[*] Unable to find correct string... :(\n");
	else
		fprintf(stdout, "[*] Found the string in %03d generations. :-)\n",g);
	fprintf(stdout, "[*] Best string: %s\n", best->s);
	fprintf(stdout,"[*] Finished!\n");
	fprintf(stdout,"[*] Everything took %0.5f seconds\n", cpuSecond() - progStart);

	destroy_population(h_pop);
	CHECK(cudaFree(d_pop));
	CHECK(cudaFree(d_dest));
	CURAND_CALL(curandDestroyGenerator(gen));

	return EXIT_SUCCESS;
}
