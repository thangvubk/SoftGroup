/*
Octree Ball Query
Written by Thang Vu
All Rights Reserved 2022.
*/
#include "octree_ball_query.h"

Octree::Octree() {}

Octree::~Octree() { delete_node(root); }

void Octree::delete_node(Node *node) {
  if (!node->is_leaf)
    for (int i = 0; i < 8; i++)
      delete_node(node->octants[i]);
  delete node;
}

void Octree::build_tree(const float *points, const float *xyzwhl,
                        int num_levels, int num_points) {
  this->points = points;
  this->num_levels = num_levels;

  // root node
  root = new Node;
  root->box =
      (Box){xyzwhl[0], xyzwhl[1], xyzwhl[2], xyzwhl[3], xyzwhl[4], xyzwhl[5]};
  root->num_points = num_points;

  // build octants
  build_octants(root);
}

// clang-format off
/*
Get octant ind of a point s.t. parent box
Octant order from a parent box
          z
          ^
          |
          4 -------- 6
         /|         /|
        5 -------- 7 |
        | |        | |
        | 0 -------| 2 -----> y
        |/         |/
        1 -------- 3
       /
      x
*/
// clang-format on
int Octree::get_octant_ind(const float *points, int pt_ind, Box box) {
  int ind_x = points[3 * pt_ind] < box.x ? 0 : 1;
  int ind_y = points[3 * pt_ind + 1] < box.y ? 0 : 1;
  int ind_z = points[3 * pt_ind + 2] < box.z ? 0 : 1;
  return (ind_z << 2) + (ind_y << 1) + ind_x;
}

// get octant box from parent box
Box Octree::get_octant_box(Box pa_box, int ind) {
  float x, y, z, w, h, l;
  w = pa_box.w / 2;
  h = pa_box.h / 2;
  l = pa_box.l / 2;
  if ((ind >> 0) & 1)
    x = pa_box.x + w / 2;
  else
    x = pa_box.x - w / 2;

  if ((ind >> 1) & 1)
    y = pa_box.y + h / 2;
  else
    y = pa_box.y - h / 2;

  if ((ind >> 2) & 1)
    z = pa_box.z + l / 2;
  else
    z = pa_box.z - l / 2;
  Box box = {x, y, z, w, h, l};
  return box;
}

// build octants for a parent node
void Octree::build_octants(Node *pa_node) {
  int level = pa_node->level + 1;
  if (level > num_levels)
    return;

  // create new octant then attach to parent node
  for (int i = 0; i < 8; i++) {
    Node *octant = new Node;
    octant->box = get_octant_box(pa_node->box, i);
    octant->level = level;
    if (octant->level == num_levels)
      octant->is_leaf = true;
    pa_node->octants[i] = octant;
  }

  // split parent pt_idx into octants ind
  for (int i = 0; i < pa_node->num_points; ++i) {
    // we dont create pt_inds for root since it naturally covers all points
    int pt_ind = pa_node->pt_inds.size() == 0 ? i : pa_node->pt_inds[i];
    int oct_ind = get_octant_ind(points, pt_ind, pa_node->box);
    pa_node->octants[oct_ind]->pt_inds.push_back(pt_ind);
    pa_node->octants[oct_ind]->num_points++;
  }

  // build next level octants
  for (int i = 0; i < 8; i++) {
    build_octants(pa_node->octants[i]);
  }
}

// export data and tree structure in breath-first order
void Octree::export_data(float *boxes, int *pt_inds, int *pt_start_len) {
  int node_ind = 0;
  int leaf_ind = 0;
  int pt_count = 0;
  std::queue<Node *> queue;
  queue.push(root);
  while (!queue.empty()) {
    Node *node = queue.front();
    queue.pop();

    // box for each node
    boxes[node_ind * 6] = node->box.x;
    boxes[node_ind * 6 + 1] = node->box.y;
    boxes[node_ind * 6 + 2] = node->box.z;
    boxes[node_ind * 6 + 3] = node->box.w;
    boxes[node_ind * 6 + 4] = node->box.h;
    boxes[node_ind * 6 + 5] = node->box.l;
    node_ind++;

    // export data on leaf node only
    if (node->is_leaf) {
      pt_start_len[leaf_ind * 2] = pt_count;
      pt_start_len[leaf_ind * 2 + 1] = node->pt_inds.size();
      leaf_ind++;
      for (int i = 0; i < (int)node->pt_inds.size(); i++)
        pt_inds[pt_count++] = node->pt_inds[i];
    }

    // enqueue octants
    if (!node->is_leaf)
      for (int i = 0; i < 8; i++)
        queue.push(node->octants[i]);
  }
}

void build_and_export_octree(const at::Tensor points_tensor,
                             const at::Tensor xyzwhl_tensor,
                             at::Tensor boxes_tensor, at::Tensor pt_inds_tensor,
                             at::Tensor pt_start_len_tensor,
                             const int num_levels) {
  const float *points = points_tensor.data_ptr<float>();
  const float *xyzwhl = xyzwhl_tensor.data_ptr<float>();
  float *boxes = boxes_tensor.data_ptr<float>();
  int *pt_inds = pt_inds_tensor.data_ptr<int>();
  int *pt_start_len = pt_start_len_tensor.data_ptr<int>();
  const int num_points = points_tensor.size(0);
  Octree *tree = new Octree;
  tree->build_tree(points, xyzwhl, num_levels, num_points);
  tree->export_data(boxes, pt_inds, pt_start_len);
  delete tree;
}

int octree_ball_query(const at::Tensor points_tensor,
                      const at::Tensor boxes_tensor,
                      const at::Tensor pt_inds_tensor,
                      const at::Tensor pt_start_len_tensor,
                      at::Tensor out_inds_tensor,
                      at::Tensor out_start_len_tensor, const int mean_active,
                      const float radius) {
  const float *points = points_tensor.data_ptr<float>();
  const float *boxes = boxes_tensor.data_ptr<float>();
  const int *pt_inds = pt_inds_tensor.data_ptr<int>();
  const int *pt_start_len = pt_start_len_tensor.data_ptr<int>();
  int *out_inds = out_inds_tensor.data_ptr<int>();
  int *out_start_len = out_start_len_tensor.data_ptr<int>();
  const int num_points = points_tensor.size(0);
  const int num_nodes = boxes_tensor.size(0);
  const int num_leaves = pt_start_len_tensor.size(0);

  int ntotals = octree_ball_query_cuda_launcher(
      points, boxes, pt_inds, pt_start_len, out_inds, out_start_len,
      mean_active, radius, num_points, num_nodes, num_leaves);
  return ntotals;
}
