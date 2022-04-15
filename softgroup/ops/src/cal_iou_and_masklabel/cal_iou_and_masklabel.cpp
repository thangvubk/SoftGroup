/*
Get the IoU between predictions and gt masks
*/

#include "cal_iou_and_masklabel.h"

void get_mask_iou_on_cluster(at::Tensor proposals_idx_tensor,
                             at::Tensor proposals_offset_tensor,
                             at::Tensor instance_labels_tensor,
                             at::Tensor instance_pointnum_tensor,
                             at::Tensor proposals_iou_tensor, int nInstance,
                             int nProposal) {
  int *proposals_idx = proposals_idx_tensor.data_ptr<int>();
  int *proposals_offset = proposals_offset_tensor.data_ptr<int>();
  long *instance_labels = instance_labels_tensor.data_ptr<long>();
  int *instance_pointnum = instance_pointnum_tensor.data_ptr<int>();
  float *proposals_iou = proposals_iou_tensor.data_ptr<float>();

  // input: nInstance (1,), int
  // input: nProposal (1,), int
  // input: proposals_idx (sumNPoint), int
  // input: proposals_offset (nProposal + 1), int
  // input: instance_labels (N), long, 0~total_nInst-1, -100
  // input: instance_pointnum (total_nInst), int
  // input: mask_scores_sigmoid (sumNPoint, 1), float
  // output: proposals_iou (nProposal, total_nInst), float
  // output: mask_label (sumNPoint, 1), float
  get_mask_iou_on_cluster_cuda(nInstance, nProposal, proposals_idx,
                               proposals_offset, instance_labels,
                               instance_pointnum, proposals_iou);
}

void get_mask_iou_on_pred(at::Tensor proposals_idx_tensor,
                          at::Tensor proposals_offset_tensor,
                          at::Tensor instance_labels_tensor,
                          at::Tensor instance_pointnum_tensor,
                          at::Tensor proposals_iou_tensor, int nInstance,
                          int nProposal,
                          at::Tensor mask_scores_sigmoid_tensor) {
  int *proposals_idx = proposals_idx_tensor.data_ptr<int>();
  int *proposals_offset = proposals_offset_tensor.data_ptr<int>();
  long *instance_labels = instance_labels_tensor.data_ptr<long>();
  int *instance_pointnum = instance_pointnum_tensor.data_ptr<int>();
  float *proposals_iou = proposals_iou_tensor.data_ptr<float>();
  float *mask_scores_sigmoid = mask_scores_sigmoid_tensor.data_ptr<float>();

  // input: nInstance (1,), int
  // input: nProposal (1,), int
  // input: proposals_idx (sumNPoint), int
  // input: proposals_offset (nProposal + 1), int
  // input: instance_labels (N), long, 0~total_nInst-1, -100
  // input: instance_pointnum (total_nInst), int
  // input: mask_scores_sigmoid (sumNPoint, 1), float
  // output: proposals_iou (nProposal, total_nInst), float
  // output: mask_label (sumNPoint, 1), float
  get_mask_iou_on_pred_cuda(
      nInstance, nProposal, proposals_idx, proposals_offset, instance_labels,
      instance_pointnum, proposals_iou, mask_scores_sigmoid);
}

void get_mask_label(at::Tensor proposals_idx_tensor,
                    at::Tensor proposals_offset_tensor,
                    at::Tensor instance_labels_tensor,
                    at::Tensor instance_cls_tensor,
                    at::Tensor proposals_iou_tensor, int nInstance,
                    int nProposal, float iou_thr,
                    at::Tensor mask_labels_tensor) {
  int *proposals_idx = proposals_idx_tensor.data_ptr<int>();
  int *proposals_offset = proposals_offset_tensor.data_ptr<int>();
  long *instance_labels = instance_labels_tensor.data_ptr<long>();
  long *instance_cls = instance_cls_tensor.data_ptr<long>();
  float *proposals_iou = proposals_iou_tensor.data_ptr<float>();
  float *mask_label = mask_labels_tensor.data_ptr<float>();

  // input: nInstance (1,), int
  // input: nProposal (1,), int
  // input: proposals_idx (sumNPoint), int
  // input: proposals_offset (nProposal + 1), int
  // input: instance_labels (N), long, 0~total_nInst-1, -100
  // input: instance_pointnum (total_nInst), int
  // input: mask_scores_sigmoid (sumNPoint, 1), float
  // output: proposals_iou (nProposal, total_nInst), float
  // output: mask_label (sumNPoint, 1), float
  get_mask_label_cuda(nInstance, nProposal, iou_thr, proposals_idx,
                      proposals_offset, instance_labels, instance_cls,
                      proposals_iou, mask_label);
}
