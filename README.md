# Introduction
Analogous to the [simpleHelloGA](https://github.com/minterciso/simpleHelloGA) I created a CUDA version of a static version of the simpleHelloGA (you can't pass a different string, you have to edit the main.cu and consts.h files for this).

This is only an example that actually runs **lower** than the simpleHelloGA C code. This is mainly due to memory copy back and from the host to the GPU. The Kernel itself is pretty fast but is also kind of serialized with the atomicAdd() function.

# How it works
Genetic Algorithms relly heavily on randomness, both for creating the initial host, crossing over, mutating, selecting the individuals, etc. In this simple *Hello World* of Genetic Algorithms we currently keep it extremelly simple, and by that we mean:

1. We **always** keep 20% of the population.
2. We **always** calculate the fitness based on the destination string, not the population of the generation.
3. We create all the random numbers on the GPU, since we have some extra benefits for doing this (one call to create all numbers, then we just browse the array).

So, that being said the overall algorithm is:

1. Initialize the CURAND library
2. Create the population on the host
3. For each generation:
    1. Copy the host population to the device
    2. Call the kernel to calculate the fitness
    3. Copy the device population to the host
    4. Sort the host based on fitness
    5. Crossover and mutate
4. Clear the memory

The Crossover and mutation stands for some more information:

1. Create random numbers of int type: n=(POP_SIZE-KEEP_POP)/2 + (POP_SIZE-KEEP_POP)
2. Create random numbers for mutation probability: n=KEEP_POP\*LEN_SIZE
3. Create random numbers for random mutation results: n=POP_SIZE\*LEN_SIZE

After this is all pretty standard:

4. Kill the low performance individuals
5. Reset the fitness of the best performance individuals
6. While population < population size:
    1. Select 2 numbers from the random int arrays, the parents
    2. Select 1 number from the random int arrays, the xover point
    3. Create 2 sons with single point crossover
    4. For each bit on the string for each son:
        1. Select number from random mutation array
        2. If number (downsized to float) is < probability of mutation:
        3. Select next number from int pool
        4. Change it to a char representation
        5. Substitute on the string
    
# Results
Running on a GTX 1060, the results are:
    
    [*] Starting Release/cuHelloGA... 
    [*] Setting up Random generators...[OK] (0.08292)
    [*] Creating population...[OK] (0.00499)
    [*] Using Kernel parameters: <<<(16,2),(32,32)>>>
    [*] Evolving...[OK] (0.48719)
    [*] Found the string in 1000 generations. :-)
    [*] Best string: Hello World from a motherf****ing perspective!
    [*] Finished!
    [*] Everything took 0.57611 seconds

That basically means that, for running 1000 generations it took 0.57611 seconds, however the evolution part took 0.48719 seconds. And what about the Kernel? Well, nvprof gives us a whopping 5.33 microSeconds in average since we are running 1000 times, we can say that it takes around 5 ms to run all kernels. And this with an occupance of onoly 0,56.

# Increasing performance
If the kernel is fast, what can we do to improve the performance?

We can do a few things, in order of importance:

1. Sort on the GPU
2. Crossover and mutate on the GPU
3. Find a way to circumvent the atomicAdd()

## Sorting, crossing and mutating on the GPU
Due to the fact that we are sorting, crossing and mutating on the CPU, for every generation (and kernel launch) we need to copy memory from host to device at least 2 times (one from device to host, and another from host to device). This is time consuming, like a lot.

In order to remove this we need to put all this on the GPU, that way we limit the copy from device to host and vice versa to only 2, one in the beginning with the first host, and one at the end with the result of the evolution.

For this we may need to think a little more in terms of device coding and paralelism (paralel sort is never easy).

Maybe we can do some tests with the Thrust library?

## Finding a way to circumventt the atomicAdd()
We basically use the atomicAdd() to move away of creating a reduction kernel (which is also nott at all simple). Maybe (just maybe, since this is such a simple display) using a reduction kernel increases the paralelism, and therefore the occupancy. If this happens, maybe we get some extra time from this.
