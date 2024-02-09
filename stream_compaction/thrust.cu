#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            int nBytes = n*sizeof(int); 
            int* g_in; 
            int* g_out; 

            cudaMalloc((void**)&g_in, nBytes); 
            cudaMalloc((void**)&g_out, nBytes); 

            cudaMemcpy(g_in, idata, nBytes, cudaMemcpyHostToDevice); 

            thrust::device_vector<int> g_in_thrust(g_in, g_in+n); 
            thrust::device_vector<int> g_out_thrust(n);  

            timer().startGpuTimer();
            thrust::exclusive_scan(g_in_thrust.begin(), g_in_thrust.end(), g_out_thrust.begin());
            timer().endGpuTimer();

            thrust::copy(g_out_thrust.begin(), g_out_thrust.end(), odata); 
            cudaFree(g_in); cudaFree(g_out); 
        }
    }
}
