/*
Get the IoU between predictions and gt masks
*/

#ifndef CAL_IOU_AND_MASKLABEL_H
#define CAL_IOU_AND_MASKLABEL_H
#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>

#include "../datatype/datatype.h"

void get_mask_iou_on_cluster_cuda(int nInstance, int nProposal,
                                  int *proposals_idx, int *proposals_offset,
                                  long *instance_labels, int *instance_pointnum,
                                  float *proposals_iou);

void get_mask_iou_on_pred_cuda(int nInstance, int nProposal, int *proposals_idx,
                               int *proposals_offset, long *instance_labels,
                               int *instance_pointnum, float *proposals_iou,
                               float *mask_scores_sigmoid);

void get_mask_label_cuda(int nInstance, int nProposal, float iou_thr,
                         int *proposals_idx, int *proposals_offset,
                         long *instance_labels, long *instance_cls,
                         float *proposals_iou, float *mask_label);

void get_mask_iou_on_cluster(at::Tensor proposals_idx_tensor,
                             at::Tensor proposals_offset_tensor,
                             at::Tensor instance_labels_tensor,
                             at::Tensor instance_pointnum_tensor,
                             at::Tensor proposals_iou_tensor, int nInstance,
                             int nProposal);

void get_mask_iou_on_pred(at::Tensor proposals_idx_tensor,
                          at::Tensor proposals_offset_tensor,
                          at::Tensor instance_labels_tensor,
                          at::Tensor instance_pointnum_tensor,
                          at::Tensor proposals_iou_tensor, int nInstance,
                          int nProposal, at::Tensor mask_scores_sigmoid_tensor);

void get_mask_label(at::Tensor proposals_idx_tensor,
                    at::Tensor proposals_offset_tensor,
                    at::Tensor instance_labels_tensor,
                    at::Tensor instance_cls_tensor,
                    at::Tensor proposals_iou_tensor, int nInstance,
                    int nProposal, float iou_thr,
                    at::Tensor mask_labels_tensor);

#endif // CAL_IOU_AND_MASKLABEL_H
