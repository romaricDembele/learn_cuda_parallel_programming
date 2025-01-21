#include "cuda_runtime.h"
#include "AddVectors.h"
#include <stdio.h>

// Dimension of the vectors to be added
#define DIM 32

#define syncCheckError(err) {syncCheck((err), __FILE__, __LINE__);}
inline void syncCheck(cudaError_t err, const char *file, int line, bool abort=true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

#define asyncCheckError() {asynchCheck(__FILE__, __LINE__);}
inline void asynchCheck(const char *file, int line, bool abort=true) {
    cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

// Add Two vectors Kernel definition
__global__ void addVectors(int* A, int* B, int* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

// ADDITION PROGRAM
void addition() {
    // long long DIM = 1024LL*1024*32;
    // Declare pointers on the HOST memory for the vectors
    // to be added (A and B) and the vector holding the result
    int* A;
    int* B;
    int* C;

    // Declare pointers on the HOST memory to hold GPU addresses 
    // of the two vectors to be added.
    int* device_A;
    int* device_B;
    int* device_C;

    // Allocate memory on the DEVICE memory heap for the tow 
    // vectors to be added
    syncCheckError(cudaMalloc((void **)&device_A, DIM * sizeof(int)));
    syncCheckError(cudaMalloc((void **)&device_B, DIM * sizeof(int)));
    syncCheckError(cudaMalloc((void **)&device_C, DIM * sizeof(int)));

    // Allocate memory on the HOST memory heap for the tow 
    // vectors to be added
    A = (int *)malloc(DIM * sizeof(int));
    B = (int *)malloc(DIM * sizeof(int));
    C = (int *)malloc(DIM * sizeof(int));

    // Initialize the vectors to be added in the HOST memory
    for (int i = 0; i < DIM; i++) {
        A[i] = i;
        B[i] = DIM - i;
    }

    // Copy the vectors to the DEVICE memory
    syncCheckError(cudaMemcpy(device_A, A, DIM * sizeof(int), cudaMemcpyHostToDevice));
    syncCheckError(cudaMemcpy(device_B, B, DIM * sizeof(int), cudaMemcpyHostToDevice));

    // Call the Addition Kernel function
    addVectors<<<1, DIM>>>(device_A, device_B, device_C);
    asyncCheckError();

    // Copy the result back to the HOST memory
    syncCheckError(cudaMemcpy(C, device_C, DIM * sizeof(DIM), cudaMemcpyDeviceToHost));
 
    // Ensure that the CPU wait for the GPU workers to finish
    // their jobs.
    cudaDeviceSynchronize();

    // Print on the standard output
    for (int i = 0; i < DIM; i++) {
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
    }

    // Free the memory both for the GPU and the CPU
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_B);
    free(A);
    free(B);
    free(C);
}