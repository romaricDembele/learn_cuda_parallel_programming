#include "cuda_runtime.h"
#include <stdio.h>

// Kernel definition
__global__ void hello_world() {
    printf("\n[Block Id = %d] -- [Warp Id = %d] -- [Thread Id = %d]\n\n", blockIdx.x, threadIdx.x / 32, threadIdx.x);
}

int main() {
    hello_world<<<1, 64>>>();
    // Ensure that the CPU wait for the GPC workers
    cudaDeviceSynchronize();
    return 0;
}