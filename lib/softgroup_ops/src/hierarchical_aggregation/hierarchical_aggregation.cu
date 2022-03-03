#include "hierarchical_aggregation.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define MAX_PRIMARY_NUM 1024
#define MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM  1024
#define INFINITY_DIS_SQUARE 10000
#define MAX_PER_PRIMARY_ABSORB_POINT_NUM 8192
#define MAX_THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


    
// input: cuda_fragment_centers (fragment_num * 5,), 5 for (x, y, z, cls_label, batch_idx)
// input: cuda_primary_centers  (primary_num * 5,), 5 for (x, y, z, cls_label, batch_idx)
// input: ...
// output: cuda_primary_absorb_fragment_idx
// output: cuda_primary_absorb_fragment_cnt
__global__ void fragment_find_primary_(int primary_num, int *cuda_primary_offsets, float *cuda_primary_centers,
    int fragment_num, int *cuda_fragment_offsets, float *cuda_fragment_centers,
    int *cuda_primary_absorb_fragment_idx, int *cuda_primary_absorb_fragment_cnt){

    int fragment_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fragment_idx >= fragment_num) return;

    // find the nearest primary for each fragment
    float nearest_dis_square = INFINITY_DIS_SQUARE;
    int nearest_idx = -1; // primary_idx
    for( int i = 0; i < primary_num; i++){
        if (abs(cuda_primary_centers[i * 5 + 3] - cuda_fragment_centers[fragment_idx * 5 + 3]) > 0.1){ //judge same cls_label or not
            continue;
        }
        if (abs(cuda_primary_centers[i * 5 + 4] - cuda_fragment_centers[fragment_idx * 5 + 4]) > 0.1){ //judge same batch_idx or not
            continue;
        }
        float temp_dis_square = pow((cuda_primary_centers[i * 5 + 0] - cuda_fragment_centers[fragment_idx * 5 + 0]), 2)
            + pow((cuda_primary_centers[i * 5 + 1] - cuda_fragment_centers[fragment_idx * 5 + 1]), 2)
            + pow((cuda_primary_centers[i * 5 + 2] - cuda_fragment_centers[fragment_idx * 5 + 2]), 2);
        if (temp_dis_square < nearest_dis_square){
            nearest_dis_square = temp_dis_square;
            nearest_idx = i;
        }
    }
    if (nearest_idx == -1) return; // fragment not belong to any primary

    // r_size
    int primary_point_num = cuda_primary_offsets[nearest_idx + 1] - cuda_primary_offsets[nearest_idx];
    float r_size = 0.01 * sqrt(float(primary_point_num));

    // r_cls
    // instance radius for each class, statistical data from the training set
    float class_radius_mean[20] = {-1., -1., 0.7047687683952325, 1.1732690381942337,  0.39644035821116036, \
        1.011516629020215,  0.7260155292902369, 0.8674973999335017, 0.8374931435447094,  1.0454153869133096, \
        0.32879464797430913,  1.1954566226966346,  0.8628817944400078,  1.0416287916782507, 0.6602697958671507,  \
        0.8541363897836871, 0.38055290598206537, 0.3011878752684007,  0.7420871812436316,  0.4474268644407741};
    int _class_idx = (int)cuda_fragment_centers[fragment_idx * 5 + 3];
    float r_cls = class_radius_mean[_class_idx] * 1.;

    // r_set
    float r_set =  max(r_size, r_cls);

    // judge
    if ( nearest_dis_square < r_set * r_set ){ 
        int _offect = atomicAdd(cuda_primary_absorb_fragment_cnt + nearest_idx, 1);
        if (_offect < MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM)
            cuda_primary_absorb_fragment_idx[nearest_idx * MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM + _offect] = fragment_idx;
        else {
            ;
        }
    }
}

// input: ...
// output: cuda_concat_idxs
// output: cuda_concat_point_num,
__global__ void concat_fragments_(
    int *cuda_fragment_idxs, int *cuda_fragment_offsets,
    int *cuda_primary_idxs, int *cuda_primary_offsets,
    int *cuda_primary_absorb_fragment_idx, int *cuda_primary_absorb_fragment_cnt,
    int *cuda_concat_idxs, int *cuda_concat_point_num,
    int primary_num){
    
    int primary_idx = blockIdx.x;
    if (primary_idx >= primary_num) return;

    int _accu_offset = 0; // unit is point
    for (int i=0; i<cuda_primary_absorb_fragment_cnt[primary_idx] && i<MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM; i++){
        int idx = cuda_primary_absorb_fragment_idx[primary_idx * MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM + i];
        for (int j=cuda_fragment_offsets[idx]; j<cuda_fragment_offsets[idx + 1]; j++){
            if (_accu_offset < MAX_PER_PRIMARY_ABSORB_POINT_NUM) {
                cuda_concat_idxs[primary_idx * MAX_PER_PRIMARY_ABSORB_POINT_NUM * 2 + _accu_offset * 2 + 0] = primary_idx;
                cuda_concat_idxs[primary_idx * MAX_PER_PRIMARY_ABSORB_POINT_NUM * 2 + _accu_offset * 2 + 1] = cuda_fragment_idxs[j * 2 + 1];
                _accu_offset++;
            }
            else {
                ;
            }
        }
    }
    cuda_concat_point_num[primary_idx] = _accu_offset;
}

void hierarchical_aggregation_cuda(
    int fragment_total_point_num, int fragment_num, int *fragment_idxs, int *fragment_offsets, float *fragment_centers,
    int primary_total_point_num, int primary_num, int *primary_idxs, int *primary_offsets, float *primary_centers,
    int *primary_idxs_post, int *primary_offsets_post){

    if (primary_num == 0){
        return;
    }
    // on devices, allocate and copy memory
    int *cuda_fragment_idxs;
    int *cuda_fragment_offsets;
    float *cuda_fragment_centers;
    cudaMalloc((void**)&cuda_fragment_idxs, fragment_total_point_num * 2 * sizeof(int) + sizeof(int)); // prevent alloc 0 space
    cudaMalloc((void**)&cuda_fragment_offsets, (fragment_num + 1) * sizeof(int));
    cudaMalloc((void**)&cuda_fragment_centers, fragment_num * 5 * sizeof(float) + sizeof(float));  // prevent alloc 0 space
    cudaMemcpy(cuda_fragment_idxs, fragment_idxs, fragment_total_point_num * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_fragment_offsets, fragment_offsets, (fragment_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_fragment_centers, fragment_centers, fragment_num * 5 * sizeof(float), cudaMemcpyHostToDevice);

    int *cuda_primary_idxs;
    int *cuda_primary_offsets;
    float *cuda_primary_centers;
    cudaMalloc((void**)&cuda_primary_idxs, primary_total_point_num * 2 * sizeof(int) + sizeof(int)); // prevent alloc 0 space
    cudaMalloc((void**)&cuda_primary_offsets, (primary_num + 1) * sizeof(int));
    cudaMalloc((void**)&cuda_primary_centers, primary_num * 5 * sizeof(float) + sizeof(float));  // prevent alloc 0 space
    cudaMemcpy(cuda_primary_idxs, primary_idxs, primary_total_point_num * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_primary_offsets, primary_offsets, (primary_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_primary_centers, primary_centers, primary_num * 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // // for each fragment, find its primary
    int *cuda_primary_absorb_fragment_idx; // array for saving the fragment idxs
    int *cuda_primary_absorb_fragment_cnt; // array for saving the fragment nums
    cudaMalloc((void**)&cuda_primary_absorb_fragment_idx, primary_num * MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM * sizeof(int) + sizeof(int));
    cudaMalloc((void**)&cuda_primary_absorb_fragment_cnt, primary_num * sizeof(int) + sizeof(int));
    if (fragment_num != 0)
        fragment_find_primary_<<<int(DIVUP(fragment_num, MAX_THREADS_PER_BLOCK)), (int)MAX_THREADS_PER_BLOCK>>>(
            primary_num, cuda_primary_offsets, cuda_primary_centers,
            fragment_num, cuda_fragment_offsets, cuda_fragment_centers,
            cuda_primary_absorb_fragment_idx, cuda_primary_absorb_fragment_cnt);
    cudaDeviceSynchronize();

    // concatenate fragments belonging to the same primary
    int *cuda_concat_idxs;
    int *cuda_concat_point_num;
    cudaMalloc((void**)&cuda_concat_idxs, primary_num * MAX_PER_PRIMARY_ABSORB_POINT_NUM * 2 * sizeof(int) + sizeof(int));
    cudaMalloc((void**)&cuda_concat_point_num, primary_num *  sizeof(int) + sizeof(int));
    assert(primary_num <= MAX_PRIMARY_NUM);
    concat_fragments_<<<primary_num, (int)1>>>(
        cuda_fragment_idxs, cuda_fragment_offsets,
        cuda_primary_idxs, cuda_primary_offsets,
        cuda_primary_absorb_fragment_idx, cuda_primary_absorb_fragment_cnt,
        cuda_concat_idxs, cuda_concat_point_num,
        primary_num);
    cudaDeviceSynchronize();

    // merge primary instances and fragments
    int *concat_point_num = new int [primary_num + 1]; // allocate on host
    cudaMemcpy(concat_point_num, cuda_concat_point_num, primary_num * sizeof(int), cudaMemcpyDeviceToHost);
    int _accu_offset = 0;
    for (int i=0; i < primary_num; i++){
        // add primary instances
        cudaMemcpy(primary_idxs_post + _accu_offset * 2, 
            cuda_primary_idxs + primary_offsets[i] * 2,
            (primary_offsets[i + 1] - primary_offsets[i]) * 2 * sizeof(int), 
            cudaMemcpyDeviceToHost);
        _accu_offset += (primary_offsets[i + 1] - primary_offsets[i]);

        // add absorbed fragments
        cudaMemcpy(primary_idxs_post + _accu_offset * 2, 
            cuda_concat_idxs + i * MAX_PER_PRIMARY_ABSORB_POINT_NUM * 2, 
            concat_point_num[i] * 2 * sizeof(int), 
            cudaMemcpyDeviceToHost);
        _accu_offset += concat_point_num[i];

        // writing offsets
        primary_offsets_post[i + 1] = _accu_offset;
    }
    cudaDeviceSynchronize();

    cudaError_t err;
    err  = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}