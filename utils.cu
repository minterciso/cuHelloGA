/*
 * utils.c
 *
 *  Created on: 16/12/2018
 *      Author: minterciso
 */
#include "utils.h"

#include <sys/time.h>
#include <stdio.h>

double cpuSecond(){
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


