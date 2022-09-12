/*
Calculate the IoU between predictions and GTs and generate mask labels
*/
#include "../cuda_utils.h"
#include "cal_iou_and_masklabel.h"
#include <math.h>
#include <stdio.h>

__global__ void
get_mask_iou_on_cluster_cuda_(int nInstance, int nProposal, int *proposals_idx,
                              int *proposals_offset, long *instance_labels,
                              int *instance_pointnum, float *proposals_iou) {

  for (int proposal_id = blockIdx.x; proposal_id < nProposal;
       proposal_id += gridDim.x) {
    int start = proposals_offset[proposal_id];
    int end = proposals_offset[proposal_id + 1];
    int proposal_total = end - start;
    for (int instance_id = threadIdx.x; instance_id < nInstance;
         instance_id += blockDim.x) {
      int instance_total = instance_pointnum[instance_id];
      int intersection = 0;
      for (int i = start; i < end; i++) {
        int idx = proposals_idx[i];
        if ((int)instance_labels[idx] == instance_id) {
          intersection += 1;
        }
      }
      proposals_iou[proposal_id * nInstance + instance_id] =
          (float)intersection /
          ((float)(proposal_total + instance_total - intersection) + 1e-5);
    }
  }
}

__global__ void
get_mask_iou_on_pred_cuda_(int nInstance, int nProposal, int *proposals_idx,
                           int *proposals_offset, long *instance_labels,
                           int *instance_pointnum, float *proposals_iou,
                           float *mask_scores_sigmoid) {

  for (int proposal_id = blockIdx.x; proposal_id < nProposal;
       proposal_id += gridDim.x) {
    int start = proposals_offset[proposal_id];
    int end = proposals_offset[proposal_id + 1];
    int proposal_total = 0;

    for (int i = start; i < end; i++)
      if (mask_scores_sigmoid[i] > 0.5)
        proposal_total += 1;

    for (int instance_id = threadIdx.x; instance_id < nInstance;
         instance_id += blockDim.x) {
      int instance_total = instance_pointnum[instance_id];
      int intersection = 0;
      for (int i = start; i < end; i++) {
        int idx = proposals_idx[i];
        if (mask_scores_sigmoid[i] > 0.5) {
          if ((int)instance_labels[idx] == instance_id)
            intersection += 1;
        }
      }
      proposals_iou[proposal_id * nInstance + instance_id] =
          (float)intersection /
          ((float)(proposal_total + instance_total - intersection) + 1e-5);
    }
  }
}

__global__ void get_mask_label_cuda_(int nInstance, int nProposal,
                                     float iou_thr, int *proposals_idx,
                                     int *proposals_offset,
                                     long *instance_labels, long *instance_cls,
                                     float *proposals_iou, float *mask_label) {
  for (int proposal_id = blockIdx.x; proposal_id < nProposal;
       proposal_id += gridDim.x) {
    int start = proposals_offset[proposal_id];
    int end = proposals_offset[proposal_id + 1];
    // int proposal_total = end - start;

    // find the instance with max iou
    float max_iou = 0.;
    int max_ind = 0;
    for (int instance_id = 0; instance_id < nInstance; instance_id++) {
      if (proposals_iou[proposal_id * nInstance + instance_id] > max_iou) {
        if (instance_cls[instance_id] != -100) { // ignored_class
          max_iou = proposals_iou[proposal_id * nInstance + instance_id];
          max_ind = instance_id;
        }
      }
    }
    // mask_label initilized with -1 (-1 means ignored)
    if (max_iou >= iou_thr) {
      for (int i = start; i < end; i++) {
        int idx = proposals_idx[i];
        if ((int)instance_labels[idx] == max_ind) {
          mask_label[i] = 1.;
        } else {
          mask_label[i] = 0.;
        }
      }
    }
  }
}

// input: nInstance (1,), int
// input: nProposal (1,), int
// input: proposals_idx (sumNPoint), int
// input: proposals_offset (nProposal + 1), int
// input: instance_labels (N), long, 0~total_nInst-1, -100
// input: instance_pointnum (total_nInst), int
// input: mask_scores_sigmoid (sumNPoint, 1), float
// output: proposals_iou (nProposal, total_nInst), float
// output: mask_label (sumNPoint, 1), float
// void cal_iou_and_masklabel_cuda(int nInstance, int nProposal, int
// *proposals_idx, int *proposals_offset, long *instance_labels, long
// *instance_cls, int *instance_pointnum, float *proposals_iou, float
// *mask_scores_sigmoid, float *mask_label, int mode){
//     get_iou_mask_cuda_<<<std::min(nProposal, (int)MAX_BLOCKS_PER_GRID),
//     std::min(nInstance, (int)MAX_THREADS_PER_BLOCK)>>>(nInstance, nProposal,
//     proposals_idx, proposals_offset, instance_labels, instance_pointnum,
//     proposals_iou, mask_scores_sigmoid, mask_label, mode);
//     cudaDeviceSynchronize();
//     get_mask_label_cuda_<<<std::min(nProposal, (int)MAX_BLOCKS_PER_GRID),
//     (int)1>>>(nInstance, nProposal, proposals_idx, proposals_offset,
//     instance_labels, instance_cls, instance_pointnum, proposals_iou,
//     mask_scores_sigmoid, mask_label);
//
// }

void get_mask_iou_on_cluster_cuda(int nInstance, int nProposal,
                                  int *proposals_idx, int *proposals_offset,
                                  long *instance_labels, int *instance_pointnum,
                                  float *proposals_iou) {
  get_mask_iou_on_cluster_cuda_<<<std::min(nProposal, (int)MAX_BLOCKS_PER_GRID),
                                  std::min(nInstance,
                                           (int)MAX_THREADS_PER_BLOCK)>>>(
      nInstance, nProposal, proposals_idx, proposals_offset, instance_labels,
      instance_pointnum, proposals_iou);
  cudaDeviceSynchronize();
}

void get_mask_iou_on_pred_cuda(int nInstance, int nProposal, int *proposals_idx,
                               int *proposals_offset, long *instance_labels,
                               int *instance_pointnum, float *proposals_iou,
                               float *mask_scores_sigmoid) {
  get_mask_iou_on_pred_cuda_<<<std::min(nProposal, (int)MAX_BLOCKS_PER_GRID),
                               std::min(nInstance,
                                        (int)MAX_THREADS_PER_BLOCK)>>>(
      nInstance, nProposal, proposals_idx, proposals_offset, instance_labels,
      instance_pointnum, proposals_iou, mask_scores_sigmoid);
  cudaDeviceSynchronize();
}

void get_mask_label_cuda(int nInstance, int nProposal, float iou_thr,
                         int *proposals_idx, int *proposals_offset,
                         long *instance_labels, long *instance_cls,
                         float *proposals_iou, float *mask_label) {
  get_mask_label_cuda_<<<std::min(nProposal, (int)MAX_BLOCKS_PER_GRID),
                         (int)1>>>(nInstance, nProposal, iou_thr, proposals_idx,
                                   proposals_offset, instance_labels,
                                   instance_cls, proposals_iou, mask_label);
  cudaDeviceSynchronize();
}
