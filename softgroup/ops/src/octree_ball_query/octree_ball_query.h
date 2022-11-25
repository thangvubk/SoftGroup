/*
Octree Ball Query
Written by Thang Vu
All Rights Reserved 2022.
*/
#ifndef OCTREE_BALL_QUERY_H
#define OCTREE_BALL_QUERY_H

#include <vector>
#include <torch/serialize/tensor.h>

struct Box {
  float x, y, z, w, h, l;
};

struct Node {
  Box box;
  int level = 0;
  int num_points = 0;
  bool is_leaf = false;
  std::vector<Int> pt_inds;
  Node *octants[8];
};

class Octree {
public:
  const float *points; // shape (n, 3)
  int num_levels;
  Node *root;

  Octree();
  ~Octree();
  void delete_node(Node *node);
  void build_tree(const float *points, const float *xyzwhl, int num_levels,
                  int num_points);
  void build_octants(Node *pa_node);
  Box get_octant_box(Box pa_box, int ind);
  int get_octant_ind(const float *points, int pt_ind, Box box);
  void export_data(float *boxes, int *pt_inds, int *pt_start_len);
};

void build_and_export_octree(const at::Tensor points_tensor,
                             const at::Tensor xyzwhl_tensor,
                             at::Tensor boxes_tensor, at::Tensor pt_inds_tensor,
                             at::Tensor pt_start_len_tensor,
                             const int num_levels);

int octree_ball_query(const at::Tensor points_tensor,
                      const at::Tensor boxes_tensor,
                      const at::Tensor pt_inds_tensor,
                      const at::Tensor pt_start_len_tensor,
                      at::Tensor out_inds_tensor, at::Tensor out_start_len,
                      const int mean_active, const float radius);

int octree_ball_query_cuda_launcher(const float *points, const float *boxes,
                                    const int *pt_inds, const int *pt_start_len,
                                    int *out_inds, int *out_start_len,
                                    const int mean_active, const float radius,
                                    const int num_points, const int num_nodes,
                                    const int num_leaves);

#endif
