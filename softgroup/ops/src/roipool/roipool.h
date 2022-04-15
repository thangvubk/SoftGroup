/*
ROI Max Pool
Written by Li Jiang
All Rights Reserved 2020.
*/

#ifndef ROIPOOL_H
#define ROIPOOL_H
#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>

#include "../datatype/datatype.h"

void global_avg_pool_fp_cuda(int nProposal, int C, float *feats,
                             int *proposals_offset, float *output_feats);

void global_avg_pool_bp_cuda(int nProposal, int C, float *d_feats,
                             int *proposals_offset, float *d_output_feats);

void global_avg_pool_fp(at::Tensor feats_tensor,
                        at::Tensor proposals_offset_tensor,
                        at::Tensor output_feats_tensor, int nProposal, int C);

void global_avg_pool_bp(at::Tensor d_feats_tensor,
                        at::Tensor proposals_offset_tensor,
                        at::Tensor d_output_feats_tensor, int nProposal, int C);

#endif // ROIPOOL_H
