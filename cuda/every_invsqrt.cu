#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> // incude the CUDA Runtime API headers
#include <cuda.h>         // include the CUDA Driver API headers

// __global__ does two things:
// in gcc, set up an external symbol for the kernel
// in nvcc, set this as a target for compilation into device code
__global__ void invsqrt_kernel(float *out, const float *X, long long size);

// we'll use a few helpers to check for errors, see the end of this file
void checkCudaError(cudaError_t err, const char *msg);
void checkCudaDriverError(CUresult res, const char *msg);
void checkHostAllocation(void *ptr, const char *msg);

int main()
{
    // expressly check for the CUDA Driver API
    CUresult res = cuInit(0);
    checkCudaDriverError(res, "CUDA Driver API initialization failed");

    // set up memory on the host
    long long size = 1LL << 10; // kilo
    size = size << 10;          // mega
    size = size << 10;          // giga
    size = 2 * size;            // 2^32 = 4 giga

    float *h_X = (float *)malloc(size * sizeof(float));
    float *h_out = (float *)malloc(size * sizeof(float));

    checkHostAllocation(h_X, "Failed to allocate host memory for input");
    checkHostAllocation(h_out, "Failed to allocate host memory for output");

    for (uint32_t i = 0; i < size; i++)
    {
        uint32_t intBits = i;
        float floatValue;
        memcpy(&floatValue, &intBits, sizeof(float));
        h_X[i] = floatValue;
    }

    // set up memory on the device
    cudaError_t status; // track status of cuda runtime API calls
    float *d_X, *d_out;

    // cudaMalloc takes a void **: generic pointer to a pointer
    status = cudaMalloc((void **)&d_X, size * sizeof(float));
    checkCudaError(status, "Failed to allocate device memory for input");

    status = cudaMalloc((void **)&d_out, size * sizeof(float));
    checkCudaError(status, "Failed to allocate device memory for output");

    status = cudaMemcpy(d_X, h_X, size * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(status, "Failed to copy input to device");

    // configure the kernel: how do we break up the work (the grid) across work units (blocks and threads)?
    int threadsPerBlock = 1024; // max out thread count, not to sleep better but because our work is dead simple
    int blocksInGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // "launch" the kernel on the GPU, go brrr
    invsqrt_kernel<<<blocksInGrid, threadsPerBlock>>>(d_out, d_X, size);

    // bring the data back onto the host
    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    // print out representative results
    printf("printing first 10 entries\n");
    for (int i = 0; i < 10; i++)
    {
        printf("invsqrt(%f) = %f\n", h_X[i], h_out[i]);
    }
    printf("printing a few interesting entries\n");
    int interesting[] = {
        1065353216, // 1
        1082130432, // 4
        1098907648, // 16
        1108344832, // 36
        1120403456  // 100
    };
    for (int i = 0; i < 5; i++)
    {
        printf("invsqrt(%f) = %f\n", h_X[interesting[i]], h_out[interesting[i]]);
    }
    printf("printing last 10 entries\n");
    for (int i = 0; i < 10; i++)
    {
        printf("invsqrt(%f) = %f\n", h_X[size - (i + 1)], h_out[size - (i + 1)]);
    }

    // clean up
    free(h_X);
    free(h_out);
    cudaFree(d_X);
    cudaFree(d_out);

    return 0;
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkHostAllocation(void *ptr, const char *msg)
{
    if (ptr == NULL)
    {
        fprintf(stderr, "%s\n", msg);
        exit(EXIT_FAILURE);
    }
}

void checkCudaDriverError(CUresult res, const char *msg)
{
    if (res != CUDA_SUCCESS)
    {
        const char *errMsg;
        cuGetErrorName(res, &errMsg);
        fprintf(stderr, "%s - %s\n", msg, errMsg);
        exit(EXIT_FAILURE);
    }
}
