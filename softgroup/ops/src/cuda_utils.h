#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cmath>

#define MAX_BLOCKS_PER_GRID 65000

#define MAX_THREADS_PER_BLOCK 1024

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
  int optimal_block_num =
      (N + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
  return min(optimal_block_num, MAX_BLOCKS_PER_GRID);
}

#endif
