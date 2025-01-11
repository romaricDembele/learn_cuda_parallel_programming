#include <stdio.h>

// helloWorld Kernel definition
__global__ void helloWorld() {
    printf("\n[Block Id = %d] -- [Warp Id = %d] -- [Thread Id = %d]\n", blockIdx.x, threadIdx.x / 32, threadIdx.x);
}

// HELLO WORLD PROGRAM
void hello() {
    helloWorld<<<1, 64>>>();
    // Ensure that the CPU wait for the GPU workers to finish
    // their jobs.
    cudaDeviceSynchronize();
}