#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream> 

#define threadPerBlock 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int N, int offset, int *g_idata){
            int index = (blockIdx.x*blockDim.x) + threadIdx.x; 
            if(index >= N) {return; }
            int k = index * offset; 
            g_idata[k + offset - 1] += g_idata[k + offset/2 - 1];
        }

        __global__ void kernDownSweep(int N, int offset, int *g_idata){
            int index = (blockIdx.x*blockDim.x) + threadIdx.x; 
            if(index >= N) {return; }
            if (N == 1) {g_idata[offset-1] = 0; }

            int k = index * offset; 
            int t = g_idata[k + offset/2 - 1]; 
            g_idata[k + offset/2 - 1] = g_idata[k + offset - 1]; 
            g_idata[k + offset - 1] += t;  
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlockPerGrid = (n + threadPerBlock - 1) / threadPerBlock; 
            int* buffer_0; // in-place operation 
            int nBytes = n * sizeof(int); 

            cudaMalloc((void**)&buffer_0, nBytes); 
            checkCUDAError("cudaMalloc buffer_0 failed"); 

            cudaMemcpy(buffer_0, idata, nBytes, cudaMemcpyHostToDevice); 
            checkCUDAError("cudaMemcpy buffer_0 failed"); 

            timer().startGpuTimer();

            int offset = 1; 
            for (auto d = 0; d<=ilog2ceil(n)-1; ++d) {
                offset = 1<<(d+1); 
                // std::cout << "offset = "<< offset << " d = " << d << " n/offset " << n/offset <<  std::endl; 
                fullBlockPerGrid = (idivjceil(n,offset) + threadPerBlock - 1) / threadPerBlock; 
                kernUpSweep<<<fullBlockPerGrid, threadPerBlock>>>(idivjceil(n,offset), offset, buffer_0); 
                 
            }
            // checkCUDAError("UP sweep "); 
            // cudaDeviceSynchronize();

            for (auto d = ilog2ceil(n)-1; d>=0; --d){
                offset = 1<<(d+1); 
                // std::cout << "offset = "<< offset << " d = " << d << " n/offset " << n/offset <<  std::endl; 
                fullBlockPerGrid = (idivjceil(n,offset) + threadPerBlock - 1) / threadPerBlock; 
                kernDownSweep<<<fullBlockPerGrid, threadPerBlock>>>(idivjceil(n,offset), offset, buffer_0); 
            }
            // checkCUDAError("DOWN sweep "); 

            timer().endGpuTimer();
            
            cudaMemcpy(odata, buffer_0, nBytes, cudaMemcpyDeviceToHost); 
            checkCUDAError("cudaMemcpy from device to host failed"); 
            cudaFree(buffer_0); 
        }
        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        
        __global__ void label_input(int n, int *idata, int *tmp){
            int index = (blockIdx.x*blockDim.x) + threadIdx.x; 
            if(index >= n) {return; }
            tmp[index] = idata[index]!=0? 1:0; 
        }

        __global__ void kern_streamCompaction(int n, int *label_scan, int *idata, int *odata, int *numElements){
            int index = (blockIdx.x*blockDim.x) + threadIdx.x; 
            if(index >= n) {return; }
            if(index==n-1) {
                *numElements = label_scan[index]; 
            }
            if (idata[index] != 0) {
                odata[label_scan[index]] = idata[index]; 
            }
        }

        __global__ void kern_pad_inputArray(int n, int *g_idata){
            int index = (blockIdx.x*blockDim.x) + threadIdx.x; 
            if(index == n) { g_idata[n] = 0;  }

        }

        int compact(int n, int *odata, const int *idata) {
            dim3 fullBlockPerGrid = (n + threadPerBlock - 1) / threadPerBlock; 
            // TODO
            odata[0] = 0; 
            int nBytes = n*sizeof(int); 
            int nBytes_pad = (n+1)*sizeof(int); 

            int *g_idata;   // pointer to a copy of idata on GPU 
            int *g_odata; 
            int *g_tmp;     // pointer to the label of array of g_idata 
            int *g_len;     // pointer to number of elements after compaction on GPU 
            int c_len; 
            
            cudaMalloc((void**)&g_idata, nBytes_pad); 
            cudaMalloc((void**)&g_odata, nBytes); 
            cudaMalloc((void**)&g_tmp, nBytes_pad); 
            cudaMalloc((void**)&g_len, sizeof(int)); 

            cudaMemcpy(g_idata, idata, nBytes, cudaMemcpyHostToDevice); 

            timer().startGpuTimer(); 

            // 0. pad input with one zero at the back
            kern_pad_inputArray<<<fullBlockPerGrid, threadPerBlock>>>(n, g_idata); 

            // 1. label the input array by if the item is non-zero 
            n += 1; 
            fullBlockPerGrid = (n + threadPerBlock - 1) / threadPerBlock; 
            label_input<<<fullBlockPerGrid, threadPerBlock>>>(n, g_idata, g_tmp); 

            // 2. compute scan of the label array  
            int offset = 1; 
            for (auto d = 0; d<=ilog2ceil(n)-1; ++d) {
                offset = 1<<(d+1); 
                // std::cout << "offset = "<< offset << " d = " << d << " n/offset " << n/offset <<  std::endl; 
                fullBlockPerGrid = (idivjceil(n,offset) + threadPerBlock - 1) / threadPerBlock; 
                kernUpSweep<<<fullBlockPerGrid, threadPerBlock>>>(idivjceil(n,offset), offset, g_tmp); 
            }
            checkCUDAError("UP sweep "); 

            for (auto d = ilog2ceil(n)-1; d>=0; --d){
                offset = 1<<(d+1); 
                // std::cout << "offset = "<< offset << " d = " << d << " n/offset " << n/offset <<  std::endl; 
                fullBlockPerGrid = (idivjceil(n,offset) + threadPerBlock - 1) / threadPerBlock; 
                kernDownSweep<<<fullBlockPerGrid, threadPerBlock>>>(idivjceil(n,offset), offset, g_tmp); 
            }
            checkCUDAError("DOWN sweep "); 

            // 3. stream compaction 
            fullBlockPerGrid = (n + threadPerBlock - 1) / threadPerBlock; 
            kern_streamCompaction<<<fullBlockPerGrid, threadPerBlock>>>(n, g_tmp, g_idata, g_odata, g_len); 

            timer().endGpuTimer();
            cudaMemcpy(odata, g_odata, nBytes, cudaMemcpyDeviceToHost); 
            cudaMemcpy(&c_len, g_len, sizeof(int), cudaMemcpyDeviceToHost); 

            cudaFree(g_idata); cudaFree(g_odata); cudaFree(g_tmp); cudaFree(g_len); 
            return c_len; 
        }
    }
}
