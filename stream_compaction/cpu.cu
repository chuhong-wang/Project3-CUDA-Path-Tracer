#include <cstdio>
#include <iostream> 
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // simple prefix-sum (scan) by a for loop 
            // exclusive prefix sum 
            odata[0] = 0; 
            for (auto i = 1; i<n; ++i){
                odata[i] = odata[i-1] + idata[i-1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // This stream compaction method will remove 0s from an array of ints.
            int curr_idx = 0; 
            for (auto i = 0; i<n ; ++i) {
                if (idata[i]!=0){
                    odata[curr_idx] = idata[i]; 
                    ++curr_idx;
                }
            }
            timer().endCpuTimer();
            return curr_idx; 
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            n+=1; // pad input array with a trailing zero 

            int *tmp = new int[n];
            int *tmp_sum = new int[n]; 

            timer().startCpuTimer();
            // label the input array by if non-zero 
            for (auto i = 0; i<n ; ++i){
                tmp[i] = idata[i]!=0? 1:0; 
            }
            tmp[n-1] = 0; 
            
            // scan the array of input label 
            tmp_sum[0] = 0; 
            for (auto i = 1; i<n; ++i){
                tmp_sum[i] = tmp_sum[i-1] + tmp[i-1];
            }

            // stream compaction  
            for (auto i = 0; i<n; ++i){
                if (tmp[i]!=0){
                    odata[tmp_sum[i]] = idata[i]; 
                }
            }
            timer().endCpuTimer();
            // for (auto i = 0; i<n ; ++i) {std::cout << tmp_sum[i] << std::endl; }
            return tmp_sum[n-1];
            free(tmp); free(tmp_sum); 
        }
    }
}
