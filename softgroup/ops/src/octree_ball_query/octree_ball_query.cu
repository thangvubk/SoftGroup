/*
Octree Ball Query
Written by Thang Vu
All Rights Reserved 2022.
*/

#include "../cuda_utils.h"
#include "octree_ball_query.h"

#define NUM_NODES 585  // 1 + 8 + 8*8 + 8*8*8
#define NUM_LEAVES 512 // 8*8*8
#define MAX_SAMPLES 1000

__device__ __forceinline__ int
is_interection(const float *box, const float *point, const float r) {
  float x = box[0], y = box[1], z = box[2], w = box[3], h = box[4], l = box[5];
  float cx = point[0], cy = point[1], cz = point[2];
  float dist_x = fabsf(x - cx);
  float dist_y = fabsf(y - cy);
  float dist_z = fabsf(z - cz);

  // not interect
  if (dist_x > (w / 2 + r))
    return 0;
  if (dist_y > (h / 2 + r))
    return 0;
  if (dist_z > (l / 2 + r))
    return 0;

  // intersect
  if (dist_x <= (w / 2))
    return 1;
  if (dist_y <= (h / 2))
    return 1;
  if (dist_z <= (l / 2))
    return 1;

  // other cases
  float delta_x = dist_x - w / 2;
  float delta_y = dist_y - h / 2;
  float delta_z = dist_z - l / 2;
  int flag = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z <= r * r;
  return flag;
}

__device__ __forceinline__ int is_neighbor(const float *p1, const float *p2,
                                           const float r) {
  float x1 = p1[0], y1 = p1[1], z1 = p1[2];
  float x2 = p2[0], y2 = p2[1], z2 = p2[2];
  float d_sqr =
      (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
  int flag = d_sqr < r * r;
  return flag;
}

__global__ void octree_ball_query_cuda(const int nthreads, const float *points,
                                       const float *boxes, const int *pt_inds,
                                       const int *pt_start_len, int *out_inds,
                                       int *out_start_len, int *ntotals,
                                       const int mean_active,
                                       const float radius, const int num_nodes,
                                       const int num_leaves) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int num_mids = NUM_NODES - NUM_LEAVES;
    int neighbor_inds[MAX_SAMPLES];
    int count = 0;
    int actives[NUM_NODES];
    for (int i = 0; i < NUM_NODES; i++)
      actives[i] = 1;
    const float *cur_point = points + index * 3;

    // iterate through all non-leaf nodes and check their octants
    for (int cur_node = 0; cur_node < num_mids; cur_node++) {
      int cur_active = actives[cur_node];
      for (int oct_offset = 0; oct_offset < 8; oct_offset++) {
        int octant = cur_node * 8 + oct_offset + 1;
        // mark all octants as inactive if their parent is inactive
        if (!cur_active) {
          actives[octant] = 0;
          continue;
        }

        const float *cur_box = boxes + octant * 6;
        int intersec_flag = is_interection(cur_box, cur_point, radius);
        // printf("octant %d intersec_flag %d\n", octant, intersec_flag);
        actives[octant] = intersec_flag;

        // if octant is an active leaf, perform nearest neighbor search
        if (intersec_flag && (octant >= num_mids)) {
          int leaf_ind = octant - (num_mids);
          int start = pt_start_len[leaf_ind * 2];
          int len = pt_start_len[leaf_ind * 2 + 1];
          int end = start + len;
          // printf("octant %d start %d len %d\n", octant, start, len);
          for (int i = start; i < end; i++) {
            int pt_ind = pt_inds[i];
            const float *oct_point = points + pt_ind * 3;
            int neighbor_flag = is_neighbor(cur_point, oct_point, radius);
            if (neighbor_flag) {
              // printf("count %d\n", count);
              if (count < MAX_SAMPLES)
                neighbor_inds[count++] = pt_ind;
              else
                break;
            }
          }
        }
      }
    }

    // append results to shared output
    out_start_len += index * 2;
    out_start_len[0] = atomicAdd(ntotals, count);
    out_start_len[1] = count;
    int num_points = nthreads;
    int thr = num_points * mean_active;
    if (out_start_len[0] >= thr)
      return;
    out_inds += out_start_len[0];
    if (out_start_len[0] + count >= thr)
      count = thr - out_start_len[0];
    for (int i = 0; i < count; i++) {
      out_inds[i] = neighbor_inds[i];
    }
  }
}

int octree_ball_query_cuda_launcher(const float *points, const float *boxes,
                                    const int *pt_inds, const int *pt_start_len,
                                    int *out_inds, int *out_start_len,
                                    const int mean_active, const float radius,
                                    const int num_points, const int num_nodes,
                                    const int num_leaves) {
  dim3 blocks(GET_BLOCKS(num_points));
  dim3 threads(MAX_THREADS_PER_BLOCK);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int ntotals = 0;
  int *p_ntotals;
  cudaMalloc((void **)&p_ntotals, sizeof(int));
  cudaMemcpy(p_ntotals, &ntotals, sizeof(int), cudaMemcpyHostToDevice);
  octree_ball_query_cuda<<<blocks, threads, 0, stream>>>(
      num_points, points, boxes, pt_inds, pt_start_len, out_inds, out_start_len,
      p_ntotals, mean_active, radius, num_nodes, num_leaves);
  cudaMemcpy(&ntotals, p_ntotals, sizeof(int), cudaMemcpyDeviceToHost);
  AT_CUDA_CHECK(cudaGetLastError());
  return ntotals;
}
