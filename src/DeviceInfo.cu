#include <stdio.h>
#include "cuda_runtime.h"
#include "DeviceInfo.h"

void getDeviceInfo() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(error));
        return;
    }

    printf("\nNumber of GPUs: %d.\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("GPU n°%d name = %s.\n", i, prop.name);
        printf("GPU n°%d threads/block max number = %d.\n", i, prop.maxThreadsPerBlock);
        
    }
    
}