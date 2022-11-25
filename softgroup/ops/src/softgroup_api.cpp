#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "softgroup_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("get_mask_iou_on_cluster", &get_mask_iou_on_cluster,
        "get_mask_iou_on_cluster");
  m.def("get_mask_iou_on_pred", &get_mask_iou_on_pred, "get_mask_iou_on_pred");
  m.def("get_mask_label", &get_mask_label, "get_mask_label");

  m.def("voxelize_idx", &voxelize_idx_3d, "voxelize_idx");
  m.def("voxelize_fp", &voxelize_fp_feat, "voxelize_fp");
  m.def("voxelize_bp", &voxelize_bp_feat, "voxelize_bp");

  m.def("build_and_export_octree", &build_and_export_octree,
        "build_and_export_octree");
  m.def("octree_ball_query", &octree_ball_query, "octree_ball_query");
  m.def("ballquery_batch_p", &ballquery_batch_p, "ballquery_batch_p");
  m.def("bfs_cluster", &bfs_cluster, "bfs_cluster");

  m.def("global_avg_pool_fp", &global_avg_pool_fp, "global_avg_pool_fp");
  m.def("global_avg_pool_bp", &global_avg_pool_bp, "global_avg_pool_bp");

  m.def("sec_mean", &sec_mean, "sec_mean");
  m.def("sec_min", &sec_min, "sec_min");
  m.def("sec_max", &sec_max, "sec_max");
}
