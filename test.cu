#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <iostream>

struct if_terminate
{
  __host__ __device__
  bool operator()(const int& x)
  {
    return x == 0;
  }
};

int main() {
    // stream compaction 
    int num_paths = 10; 
    int dev_paths[num_paths]; // Changed from int *dev_path[num_paths];
    for (auto i = 0; i<num_paths; ++i){
        dev_paths[i] = i;  // Changed from dev_path[i] = i;
    }

    auto dev_path_end = thrust::remove_if(dev_paths, dev_paths + num_paths, if_terminate()); // Corrected variable name dev_paths
    num_paths = dev_path_end - dev_paths; 

    std::cout << "remaining elements " << num_paths << std::endl; 

    return 0; 
}
