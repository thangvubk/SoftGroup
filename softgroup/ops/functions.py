import torch
from torch.autograd import Function

from . import ops


def ball_query(coords, batch_idxs, batch_offsets, radius, mean_active, with_octree=False):
    if with_octree:
        return octree_ball_query(coords, mean_active, radius)
    else:
        return ballquery_batch_p(coords, batch_idxs, batch_offsets, radius, mean_active)


def octree_ball_query(coords, mean_active, radius):
    coords = coords.cpu()
    assert coords.is_contiguous()
    xyz_max = coords.max(0)[0]
    xyz_min = coords.min(0)[0]
    xyzwhl = torch.cat([(xyz_max + xyz_min) / 2, xyz_max - xyz_min])

    n = coords.size(0)
    pt_inds = torch.zeros(coords.size(0), dtype=torch.int32)
    # TODO remove these vars
    num_nodes = 1 + 8 + 64 + 512
    num_leaves = 512
    num_levels = 3
    boxes = torch.zeros((num_nodes, 6), dtype=torch.float32)
    pt_start_len = torch.zeros((num_leaves, 2), dtype=torch.int32)
    ops.build_and_export_octree(coords, xyzwhl, boxes, pt_inds, pt_start_len, num_levels)
    boxes = boxes.cuda()
    pt_inds = pt_inds.cuda()
    pt_start_len = pt_start_len.cuda()
    coords = coords.cuda()
    while True:
        out_inds = torch.zeros(n * mean_active, dtype=torch.int32, device='cuda')
        out_start_len = torch.zeros((n, 2), dtype=torch.int32, device='cuda')
        n_totals = ops.octree_ball_query(coords, boxes, pt_inds, pt_start_len, out_inds,
                                         out_start_len, mean_active, radius)
        if n_totals <= n * mean_active:
            break
        mean_active = int(n_totals // n + 1)
    out_inds = out_inds[:n_totals]

    return out_inds, out_start_len


class GetMaskIoUOnCluster(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param instance_pointnum: (total_nInst), int
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        '''

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        proposals_iou = torch.cuda.FloatTensor(nProposal, nInstance).zero_()

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda

        ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, proposals_iou, nInstance, nProposal)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_iou_on_cluster = GetMaskIoUOnCluster.apply


class GetMaskIoUOnPred(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum,
                mask_scores_sigmoid):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param instance_pointnum: (total_nInst), int
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        '''

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        proposals_iou = torch.cuda.FloatTensor(nProposal, nInstance).zero_()

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda
        assert mask_scores_sigmoid.is_contiguous() and mask_scores_sigmoid.is_cuda

        ops.get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                 instance_pointnum, proposals_iou, nInstance, nProposal,
                                 mask_scores_sigmoid)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_iou_on_pred = GetMaskIoUOnPred.apply


class GetMaskLabel(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_cls,
                instance_pointnum, proposals_iou, iou_thr):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        '''

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        mask_label = torch.cuda.FloatTensor(proposals_idx.shape).zero_() - 1.

        assert proposals_iou.is_contiguous() and proposals_iou.is_cuda
        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_cls.is_contiguous() and instance_cls.is_cuda

        ops.get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                           proposals_iou, nInstance, nProposal, iou_thr, mask_label)

        return mask_label

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_label = GetMaskLabel.apply


class Voxelization_Idx(Function):

    @staticmethod
    def forward(ctx, coords, batchsize, mode=4):
        '''
        :param ctx:
        :param coords:  long (N, dimension + 1) or (N, dimension) dimension = 3
        :param batchsize
        :param mode: int 4=mean
        :param dimension: int
        :return: output_coords:  long (M, dimension + 1) (M <= N)
        :return: output_map: int M * (maxActive + 1)
        :return: input_map: int N
        '''
        assert coords.is_contiguous()
        N = coords.size(0)
        output_coords = coords.new()

        input_map = torch.IntTensor(N).zero_()
        output_map = input_map.new()

        ops.voxelize_idx(coords, output_coords, input_map, output_map, batchsize, mode)
        return output_coords, input_map, output_map

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None


voxelization_idx = Voxelization_Idx.apply


class Voxelization(Function):

    @staticmethod
    def forward(ctx, feats, map_rule, mode=4):
        '''
        :param ctx:
        :param map_rule: cuda int M * (maxActive + 1)
        :param feats: cuda float N * C
        :return: output_feats: cuda float M * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        N, C = feats.size()
        M = map_rule.size(0)
        maxActive = map_rule.size(1) - 1

        output_feats = torch.cuda.FloatTensor(M, C).zero_()

        ctx.for_backwards = (map_rule, mode, maxActive, N)

        ops.voxelize_fp(feats, output_feats, map_rule, mode, M, maxActive, C)
        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()

        d_feats = torch.cuda.FloatTensor(N, C).zero_()

        ops.voxelize_bp(d_output_feats.contiguous(), d_feats, map_rule, mode, M, maxActive, C)
        return d_feats, None, None


voxelization = Voxelization.apply


class BallQueryBatchP(Function):

    @staticmethod
    def forward(ctx, coords, batch_idxs, batch_offsets, radius, meanActive):
        '''
        :param ctx:
        :param coords: (n, 3) float
        :param batch_idxs: (n) int
        :param batch_offsets: (B+1) int
        :param radius: float
        :param meanActive: int
        :return: idx (nActive), int
        :return: start_len (n, 2), int
        '''

        n = coords.size(0)

        assert coords.is_contiguous() and coords.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert batch_offsets.is_contiguous() and batch_offsets.is_cuda

        while True:
            idx = torch.cuda.IntTensor(n * meanActive).zero_()
            start_len = torch.cuda.IntTensor(n, 2).zero_()
            nActive = ops.ballquery_batch_p(coords, batch_idxs, batch_offsets, idx, start_len, n,
                                            meanActive, radius)
            if nActive <= n * meanActive:
                break
            meanActive = int(nActive // n + 1)
        idx = idx[:nActive]

        return idx, start_len

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None


ballquery_batch_p = BallQueryBatchP.apply


class BFSCluster(Function):

    @staticmethod
    def forward(ctx, cluster_numpoint_mean, ball_query_idxs, start_len, threshold, class_id):
        '''
        :param ctx:
        :param ball_query_idxs: (nActive), int
        :param start_len: (N, 2), int
        :return: cluster_idxs:  int (sumNPoint, 2), dim 0 for cluster_id, dim 1 for point idxs in N
        :return: cluster_offsets: int (nCluster + 1)
        '''

        N = start_len.size(0)
        assert cluster_numpoint_mean.is_contiguous()
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        cluster_idxs = ball_query_idxs.new()
        cluster_offsets = ball_query_idxs.new()

        ops.bfs_cluster(cluster_numpoint_mean, ball_query_idxs, start_len, cluster_idxs,
                        cluster_offsets, N, threshold, class_id)

        return cluster_idxs, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        return None


bfs_cluster = BFSCluster.apply


class GlobalAvgPool(Function):

    @staticmethod
    def forward(ctx, feats, proposals_offset):
        '''
        :param ctx:
        :param feats: (sumNPoint, C) float
        :param proposals_offset: (nProposal + 1) int
        :return: output_feats (nProposal, C) float
        '''
        nProposal = proposals_offset.size(0) - 1
        sumNPoint, C = feats.size()

        assert feats.is_contiguous()
        assert proposals_offset.is_contiguous()

        output_feats = torch.cuda.FloatTensor(nProposal, C).zero_()

        ops.global_avg_pool_fp(feats, proposals_offset, output_feats, nProposal, C)

        ctx.for_backwards = (proposals_offset, sumNPoint)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        nProposal, C = d_output_feats.size()

        proposals_offset, sumNPoint = ctx.for_backwards

        d_feats = torch.cuda.FloatTensor(sumNPoint, C).zero_()

        ops.global_avg_pool_bp(d_feats, proposals_offset, d_output_feats.contiguous(), nProposal, C)

        return d_feats, None


global_avg_pool = GlobalAvgPool.apply


class SecMean(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        ops.sec_mean(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_mean = SecMean.apply


class SecMin(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        ops.sec_min(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_min = SecMin.apply


class SecMax(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        ops.sec_max(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_max = SecMax.apply
