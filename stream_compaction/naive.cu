#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define threadPerBlock 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scanGPUkernel(int offset, int N, int *buffer_d_0, int *buffer_d_1) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) {return; }
            // int offset = 1<<(d-1); 
            if (index >= offset){
                buffer_d_1[index] = buffer_d_0[index - offset] + buffer_d_0[index]; 
            }
            else {
                buffer_d_1[index] = buffer_d_0[index]; 
            }
        }

        __global__ void kern_shiftInclusive2Exclusive(int N, int *buffer_d_0, int *buffer_d_1){
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N ) {return; }
            if (index == 0) {buffer_d_1[index] = 0; }
            else {buffer_d_1[index] = buffer_d_0[index-1]; }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            dim3 fullBlockPerGrid = (n + threadPerBlock - 1) / threadPerBlock; 
            int nBytes = n * sizeof(int); 

            // printf("Size of memory to allocate: %d bytes\n", nBytes);

            int *buffer_d_0;
            int *buffer_d_1; 
            cudaMalloc((void**)&buffer_d_0, nBytes);
            checkCUDAError("cudaMalloc buffer_d_0 failed!");
            cudaMalloc((void**)&buffer_d_1, nBytes);
            checkCUDAError("cudaMalloc buffer_d_1 failed!");

            // bool output_at_1 = true;
            cudaMemcpy(buffer_d_0, idata, nBytes, cudaMemcpyHostToDevice); 
            timer().startGpuTimer(); 
            for (auto i = 1; i<=ilog2ceil(n); ++i){
                int offset = 1<<(i-1);
                // printf("i = %i ", i); 
                // printf("offset = %i \n", offset); 
                scanGPUkernel<<<fullBlockPerGrid, threadPerBlock>>>(offset, n, buffer_d_0, buffer_d_1);
                
                auto tmp = buffer_d_0; 
                buffer_d_0 = buffer_d_1; 
                buffer_d_1 = tmp;
            }
            // buffer_d_1[0] = 0; 
            kern_shiftInclusive2Exclusive<<<fullBlockPerGrid, threadPerBlock>>>(n, buffer_d_0, buffer_d_1);
            timer().endGpuTimer(); 
            cudaMemcpy(odata, buffer_d_1, nBytes, cudaMemcpyDeviceToHost); 
            cudaFree(buffer_d_0); cudaFree(buffer_d_1); 
            
        }
    }
}
