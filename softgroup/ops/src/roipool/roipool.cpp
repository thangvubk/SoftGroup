/*
ROI Max Pool
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "roipool.h"

void global_avg_pool_fp(at::Tensor feats_tensor,
                        at::Tensor proposals_offset_tensor,
                        at::Tensor output_feats_tensor, int nProposal, int C) {
  float *feats = feats_tensor.data_ptr<float>();
  int *proposals_offset = proposals_offset_tensor.data_ptr<int>();
  float *output_feats = output_feats_tensor.data_ptr<float>();

  global_avg_pool_fp_cuda(nProposal, C, feats, proposals_offset, output_feats);
}

void global_avg_pool_bp(at::Tensor d_feats_tensor,
                        at::Tensor proposals_offset_tensor,
                        at::Tensor d_output_feats_tensor, int nProposal,
                        int C) {
  float *d_feats = d_feats_tensor.data_ptr<float>();
  int *proposals_offset = proposals_offset_tensor.data_ptr<int>();
  float *d_output_feats = d_output_feats_tensor.data_ptr<float>();

  global_avg_pool_bp_cuda(nProposal, C, d_feats, proposals_offset,
                          d_output_feats);
}
