/*
Hierarchichal Aggregation Algorithm
*/

#ifndef HIERARCHICAL_AGGREGATION_H
#define HIERARCHICAL_AGGREGATION_H
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>

#include "../datatype/datatype.h"

void hierarchical_aggregation(
    at::Tensor class_numpoint_mean_tensor, at::Tensor semantic_label_tensor,
    at::Tensor coord_shift_tensor, at::Tensor batch_idxs_tensor,
    at::Tensor ball_query_idxs_tensor, at::Tensor start_len_tensor,
    at::Tensor fragment_idxs_tensor, at::Tensor fragment_offsets_tensor,
    at::Tensor fragment_centers_tensor, at::Tensor cluster_idxs_kept_tensor,
    at::Tensor cluster_offsets_kept_tensor,
    at::Tensor cluster_centers_kept_tensor, at::Tensor primary_idxs_tensor,
    at::Tensor primary_offsets_tensor, at::Tensor primary_centers_tensor,
    at::Tensor primary_idxs_post_tensor, at::Tensor primary_offsets_post_tensor,
    const int N, const int training_mode_, const int using_set_aggr_,
    const int class_id);

void hierarchical_aggregation_cuda(
    int fragment_total_point_num, int fragment_num, int *fragment_idxs,
    int *fragment_offsets, float *fragment_centers, int primary_total_point_num,
    int primary_num, int *primary_idxs, int *primary_offsets,
    float *primary_centers, int *primary_idxs_post, int *primary_offsets_post);
#endif // HIERARCHICAL_AGGREGATION_H
