/*
ROI Max Pool
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "roipool.h"
#include <math.h>
#include <stdio.h>

// fp
__global__ void global_avg_pool_fp_cuda_(int nProposal, int C, float *feats,
                                         int *proposals_offset,
                                         float *output_feats) {
  for (int pp_id = blockIdx.x; pp_id < nProposal; pp_id += gridDim.x) {
    int start = proposals_offset[pp_id];
    int end = proposals_offset[pp_id + 1];
    int n_points = end - start;

    for (int plane = threadIdx.x; plane < C; plane += blockDim.x) {
      // int argmax_idx = -1;
      // float max_val = -1e50;
      float val = 0;

      for (int i = start; i < end; i++) {
        val += feats[i * C + plane];
      }
      // output_maxidx[pp_id * C + plane] = argmax_idx;
      output_feats[pp_id * C + plane] = val / (float)n_points;
    }
  }
}

// input: feats (sumNPoint, C) float
// input: proposals_offset (nProposal + 1) int
// output: output_feats (nProposal, C) float
// output: output_maxidx (nProposal, C) int
void global_avg_pool_fp_cuda(int nProposal, int C, float *feats,
                             int *proposals_offset, float *output_feats) {
  global_avg_pool_fp_cuda_<<<std::min(nProposal, (int)32768),
                             std::min(C, (int)32)>>>(
      nProposal, C, feats, proposals_offset, output_feats);
}

// bp
__global__ void global_avg_pool_bp_cuda_(int nProposal, int C, float *d_feats,
                                         int *proposals_offset,
                                         float *d_output_feats) {
  for (int pp_id = blockIdx.x; pp_id < nProposal; pp_id += gridDim.x) {
    int start = proposals_offset[pp_id];
    int end = proposals_offset[pp_id + 1];
    int n_points = end - start;
    for (int plane = threadIdx.x; plane < C; plane += blockDim.x) {
      for (int i = start; i < end; i++) {
        atomicAdd(&d_feats[i * C + plane],
                  d_output_feats[pp_id * C + plane] / (float)n_points);
      }
    }
  }
}

// input: d_output_feats (nProposal, C) float
// input: output_maxidx (nProposal, C) int
// input: proposals_offset (nProposal + 1) int
// output: d_feats (sumNPoint, C) float
void global_avg_pool_bp_cuda(int nProposal, int C, float *d_feats,
                             int *proposals_offset, float *d_output_feats) {
  global_avg_pool_bp_cuda_<<<std::min(nProposal, (int)32768),
                             std::min(C, (int)32)>>>(
      nProposal, C, d_feats, proposals_offset, d_output_feats);
}
