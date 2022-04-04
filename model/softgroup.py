import torch
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')

from lib.softgroup_ops.functions import softgroup_ops
from util import utils
import torch.nn.functional as F
from .blocks import ResidualBlock, UBlock


class SoftGroup(nn.Module):
    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 ignore_label=-100,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 test_cfg=None,
                 fixed_modules=[],
                 pretrained=None):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.ignore_label = ignore_label
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.test_cfg = test_cfg

        # self.score_scale = cfg.score_scale
        # self.score_spatial_shape = cfg.score_spatial_shape
        # self.score_mode = cfg.score_mode

        # self.prepare_epochs = cfg.prepare_epochs
        # self.pretrain_path = cfg.pretrain_path
        # self.pretrain_module = cfg.pretrain_module
        # self.fix_module = cfg.fix_module

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(6, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(
            norm_fn(channels),
            nn.ReLU()
        )

        # semantic segmentation branch
        self.semantic_linear = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            norm_fn(channels),
            nn.ReLU(),
            nn.Linear(channels, semantic_classes)
        )

        # center shift vector branch
        self.offset_linear = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            norm_fn(channels),
            nn.ReLU(),
            nn.Linear(channels, 3, bias=True)
        )

        # topdown refinement path
        self.intra_ins_unet = UBlock([channels, 2*channels], norm_fn, 2, block, indice_key_id=11)
        self.intra_ins_outputlayer = spconv.SparseSequential(
            norm_fn(channels),
            nn.ReLU()
        )
        self.cls_linear = nn.Linear(channels, instance_classes + 1)
        self.mask_linear = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, instance_classes + 1))
        self.score_linear = nn.Linear(channels, instance_classes + 1)

        self.apply(self.set_bn_init)
        nn.init.normal_(self.score_linear.weight, 0, 0.01)
        nn.init.constant_(self.score_linear.bias, 0)

        for mod in fixed_modules:
            mod = getattr(self, mod)
            mod.eval()
            for param in mod.parameters():
                param.requires_grad = False

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(batch)
        else:
            return self.forward_test(batch)


    def forward_test(self, batch):
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), long, cuda
        instance_cls = batch['instance_cls'].cuda()            # (total_nInst), int, cuda
        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda
        spatial_shape = batch['spatial_shape']

        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = softgroup_ops.voxelization(feats, v2p_map)  # (M, C), float, cuda

        if self.test_cfg.x4_split:
            input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, 4)
            batch_idxs = torch.zeros_like(coords[:, 0].int())
        else:
            input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, 1)
            batch_idxs = coords[:, 0].int()

        semantic_scores, pt_offsets, output_feats = self.forward_backbone(
            input, p2v_map, x4_split=self.test_cfg.x4_split)  # TODO check name for map

        proposals_idx, proposals_offset = self.forward_grouping(
            semantic_scores, pt_offsets, batch_idxs, coords_float, self.grouping_cfg)

        scores_batch_idxs, cls_scores, scores, mask_scores = self.forward_instance(
            proposals_idx, proposals_offset, output_feats, coords_float)

        # scores_batch_idxs, proposals_idx, proposals_offset, mask_scores = preds['proposals']
        N = coords.size(0)
        semantic_pred = semantic_scores.max(1)[1]
        cls_scores = cls_scores.softmax(1)
        slice_inds = torch.arange(cls_scores.size(0), dtype=torch.long, device=cls_scores.device)
        cls_scores_new, cls_pred = cls_scores[:, :-1].max(1)

        cluster_scores_list = []
        clusters_list = []
        cluster_semantic_id_list = []
        semantic_label_idx = torch.arange(18) + 1
        for i in range(18):
            # arg_score = cls_pred == i
            score_inds = (cls_scores[:, i] > 0.001)
            cls_scores_new = cls_scores[:, i]
            scores_pred = scores[slice_inds, i]
            scores_pred = scores_pred.clamp(0, 1) * cls_scores_new
            # scores_pred = cls_scores_new
            # mask_cls_pred = cls_pred[scores_batch_idxs.long()]
            mask_slice_inds = torch.arange(scores_batch_idxs.size(0), dtype=torch.long, device=scores_batch_idxs.device)
            mask_scores_new = mask_scores[:, i]
            # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) 
            # (nProposal, N), int, cuda
            
            # outlier filtering
            test_mask_score_thre = -0.5 # TODO
            _mask = mask_scores_new > test_mask_score_thre
            proposals_pred[proposals_idx[_mask][:, 0].long(), proposals_idx[_mask][:, 1].long()] = 1

            # bg filtering
            # import pdb; pdb.set_trace()
            # pos_inds = (cls_pred != cfg.classes - 2)
            # proposals_pred = proposals_pred[pos_inds]
            # scores_pred = scores_pred[pos_inds]
            # cls_pred = cls_pred[pos_inds]
            # import pdb; pdb.set_trace()
            semantic_id = cls_scores.new_full(cls_scores_new.size(), semantic_label_idx[i], dtype=torch.long)


            # semantic_id1 = torch.tensor(semantic_label_idx, device=scores_pred.device) \
            #     [semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long
            # semantic_id_idx = semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]

            proposals_pred = proposals_pred[score_inds]
            scores_pred = scores_pred[score_inds]
            semantic_id = semantic_id[score_inds]

            # score threshold
            score_mask = (scores_pred > -1)
            scores_pred = scores_pred[score_mask]
            proposals_pred = proposals_pred[score_mask]
            semantic_id = semantic_id[score_mask]
            # semantic_id_idx = semantic_id_idx[score_mask]

            # npoint threshold
            proposals_pointnum = proposals_pred.sum(1)
            npoint_mask = (proposals_pointnum >= 100)
            scores_pred = scores_pred[npoint_mask]
            proposals_pred = proposals_pred[npoint_mask]
            semantic_id = semantic_id[npoint_mask]

            clusters = proposals_pred
            cluster_scores = scores_pred
            cluster_semantic_id = semantic_id

            clusters_list.append(clusters)
            cluster_scores_list.append(cluster_scores)
            cluster_semantic_id_list.append(cluster_semantic_id)
        clusters = torch.cat(clusters_list).cpu().numpy()
        cluster_scores = torch.cat(cluster_scores_list).cpu().numpy()
        cluster_semantic_id = torch.cat(cluster_semantic_id_list).cpu().numpy()
        # import pdb; pdb.set_trace()

        nclusters = clusters.shape[0]

        ret = {}
        det_ins = []
        for i in range(nclusters):
            pred = {}
            pred['scan_id'] = batch['id'][0]
            pred['conf'] = cluster_scores[i]
            pred['label_id'] = cluster_semantic_id[i]
            pred['pred_mask'] = clusters[i]
            det_ins.append(pred)
        labels = labels - 2 + 1
        labels[labels <0] = 0
        instance_labels += 1
        instance_labels[instance_labels == -99] = 0
        gt_ins = labels * 1000 + instance_labels

        gt_ins = gt_ins.cpu().numpy() 

        ret['det_ins'] = det_ins
        ret['gt_ins'] = gt_ins

        return ret


    def forward_backbone(self, input, input_map, x4_split=False):
        if x4_split:
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
            coords = self.merge_4_parts(coords)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]

        semantic_scores = self.semantic_linear(output_feats)
        semantic_scores = semantic_scores.softmax(dim=-1)
        semantic_preds = semantic_scores.max(1)[1]
        pt_offsets = self.offset_linear(output_feats)
        return semantic_scores, pt_offsets, output_feats

    def forward_4_parts(self, x, input_map):
        # helper function for s3dis: devide and forward 4 parts of a scene
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]

    def merge_4_parts(self, x):
        # helper function for s3dis: take output of 4 parts and merge them
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]
        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new

    def forward_grouping(self, semantic_scores, pt_offsets, batch_idxs, coords_float, grouping_cfg=None):
        thr = 0.2  #TODO
        proposals_idx_list = []
        proposals_offset_list = []
        batch_size = batch_idxs.max() + 1
        semantic_preds = semantic_scores.max(1)[1]  # TODO remove this

        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        class_numpoint_mean = torch.tensor(self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
        training_mode = None # TODO remove this
        for class_id in range(self.semantic_classes):
            # ignore "floor" and "wall"
            if class_id < 2:
                continue

            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > thr).nonzero().view(-1)
            if object_idxs.size(0) < 100:  # TODO
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]  # (N_fg, 3), float32

            semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()


            idx, start_len = softgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, \
                batch_idxs_, batch_offsets_, radius, mean_active)
            
            using_set_aggr = False  #TODO refactor this
            proposals_idx, proposals_offset = softgroup_ops.hierarchical_aggregation(
                class_numpoint_mean, semantic_preds_cpu, (coords_ + pt_offsets_).cpu(), idx.cpu(), start_len.cpu(),
                batch_idxs_.cpu(), training_mode, using_set_aggr, class_id)             

            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            # import pdb; pdb.set_trace()
            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)
        proposals_idx = torch.cat(proposals_idx_list, dim=0)
        proposals_offset = torch.cat(proposals_offset_list)
        return proposals_idx, proposals_offset

    def forward_instance(self, proposals_idx, proposals_offset, output_feats, coords_float):
        # proposals voxelization again
        input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords_float, **self.instance_voxel_cfg)

        # predict instance scores
        score = self.intra_ins_unet(input_feats)
        score = self.intra_ins_outputlayer(score)

        # predict mask scores
        mask_scores = self.mask_linear(score.features)
        mask_scores = mask_scores[inp_map.long()]
        scores_batch_idxs = score.indices[:, 0][inp_map.long()]

        # predict instance scores
        score_feats = self.global_pool(score)
        cls_scores = self.cls_linear(score_feats)
        iou_scores = self.score_linear(score_feats)
        
        return scores_batch_idxs, cls_scores, iou_scores, mask_scores


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, scale, spatial_shape):
        '''
        :param clusters_idx: (SumNPoint, 2), int, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = softgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = softgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = softgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / spatial_shape).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < spatial_shape)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = softgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = softgroup_ops.voxelization(clusters_feats, out_map.cuda())  # (M, C), float, cuda

        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map

    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = softgroup_ops.global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x



    def forward_old(self, input, input_map, coords, batch_idxs, batch_offsets, epoch, training_mode, gt_instances=None, split=False, semantic_only=False):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        if split:
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
            coords = self.merge_4_parts(coords)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]

        semantic_scores = self.semantic_linear(output_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long
        pt_offsets = self.offset_linear(output_feats)  # (N, 3)

        ret['semantic_scores'] = semantic_scores
        ret['pt_offsets'] = pt_offsets

        if(epoch > self.prepare_epochs) and not semantic_only:
            thr = self.cfg.score_thr
            semantic_scores = semantic_scores.softmax(dim=-1)
            proposals_idx_list = []
            proposals_offset_list = []
            cls_pred_list = []
            for class_id in range(self.cfg.semantic_classes):
                # ignore "floor" and "wall"
                if class_id < 2:
                    continue

                scores = semantic_scores[:, class_id].contiguous()
                object_idxs = (scores > thr).nonzero().view(-1)
                if object_idxs.size(0) < self.cfg.TEST_NPOINT_THRESH:
                    continue
                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = utils.get_batch_offsets(batch_idxs_, self.cfg.batch_size)
                coords_ = coords[object_idxs]
                pt_offsets_ = pt_offsets[object_idxs]  # (N_fg, 3), float32

                semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

                idx, start_len = softgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, \
                    batch_idxs_, batch_offsets_, self.point_aggr_radius, self.cluster_shift_meanActive)
                
                using_set_aggr = False  #TODO refactor this
                class_numpoint_mean = torch.tensor(self.cfg.class_numpoint_mean, dtype=torch.float32)
                proposals_idx, proposals_offset = softgroup_ops.hierarchical_aggregation(
                    class_numpoint_mean, semantic_preds_cpu, (coords_ + pt_offsets_).cpu(), idx.cpu(), start_len.cpu(),
                    batch_idxs_.cpu(), training_mode, using_set_aggr, class_id)             

                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

                # import pdb; pdb.set_trace()
                # merge proposals
                cls_pred = proposals_offset.new_full((proposals_offset.size(0) - 1,), class_id)
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]
                if proposals_idx.size(0) > 0:
                    proposals_idx_list.append(proposals_idx)
                    proposals_offset_list.append(proposals_offset)
                    cls_pred_list.append(cls_pred)

            # add gt_instances to proposals
            if gt_instances is not None:
                indices = gt_instances[:, 0]
                batch_counts = torch.bincount(indices)
                gt_instances_offset = torch.cumsum(batch_counts, dim=0)
                gt_instances_offset += proposals_offset_list[-1][-1]
                gt_instances[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_idx_list.append(gt_instances.cpu().int())
                proposals_offset_list.append(gt_instances_offset.cpu().int())
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
            cls_pred = torch.cat(cls_pred_list)

    

            # restrict the num of training proposals, avoid OOM
            max_proposal_num = getattr(self.cfg, 'max_proposal_num', 200)
            if training_mode == 'train' and proposals_offset.shape[0] > max_proposal_num:
                proposals_offset = proposals_offset[:max_proposal_num + 1]
                proposals_idx = proposals_idx[: proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]
                print('selected proposal num', proposals_offset.shape[0] - 1)

            # proposals voxelization again
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords, self.score_spatial_shape, self.score_scale, self.score_mode)

            # predict instance scores
            score = self.intra_ins_unet(input_feats)
            score = self.intra_ins_outputlayer(score)
            # score_feats = score.features[inp_map.long()] # (sumNPoint, C)

            # predict mask scores
            # first linear than voxel to point,  more efficient  (because voxel num < point num)
            mask_scores = self.mask_linear(score.features)
            mask_scores = mask_scores[inp_map.long()]
            scores_batch_idxs = score.indices[:, 0][inp_map.long()]

            # predict instance scores
            # if getattr(self.cfg, 'use_mask_filter_score_feature', False)  and \
            #         epoch > self.cfg.use_mask_filter_score_feature_start_epoch:
            #     mask_index_select = torch.ones_like(mask_scores)
            #     mask_index_select[torch.sigmoid(mask_scores) < self.cfg.mask_filter_score_feature_thre] = 0.
            #     score_feats = score_feats * mask_index_select
            # score_feats = softgroup_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
            # score_feats = softgroup_ops.global_avg_pool(score_feats, proposals_offset.cuda())
            score_feats = self.global_pool(score)
            cls_scores = self.cls_linear(score_feats)
            scores = self.score_linear(score_feats)  # (nProposal, 1)
            
            ret['proposal_scores'] = (scores_batch_idxs, cls_scores, scores, proposals_idx, proposals_offset, mask_scores)

        return ret


def model_fn_decorator(test=False):
    # config
    from util.config import cfg


    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def get_gt_instances(labels, instance_labels):
        instance_pointnum = []   # (nInst), int
        gt_cls = []
        gt_instances = []
        instance_num = int(instance_labels.max()) + 1
        inst_count = 0
        for i in range(instance_num):
            inst_idx_i = (instance_labels == i).nonzero().view(-1)
            cls_loc = inst_idx_i[0]
            cls = labels[cls_loc]
            if cls != cfg.ignore_label:
                gt_cls.append(cls)
                pad = torch.ones_like(inst_idx_i) * inst_count
                instance = torch.stack([pad, inst_idx_i], dim=1)
                gt_instances.append(instance)
                inst_count += 1
        gt_instances = torch.cat(gt_instances)
        return gt_cls, gt_instances

    def test_model_fn(batch, model, epoch, semantic_only=False):
        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()              # (N, C), float32, cuda
        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda
        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        voxel_feats = softgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        if cfg.dataset == 'scannetv2':
            input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, 1)

            ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, 'test', semantic_only=semantic_only)
        elif cfg.dataset == 's3dis':
            input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, 4)
            batch_idxs = torch.zeros_like(coords[:, 0].int())
            ret = model(input_, p2v_map, coords_float, batch_idxs, batch_offsets, epoch, 'test', split=True, semantic_only=semantic_only)
        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda

        if (epoch > cfg.prepare_epochs) and not semantic_only:
            scores_batch_idxs, cls_scores, scores, proposals_idx, proposals_offset, mask_scores = ret['proposal_scores']

        # preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs) and not semantic_only:
                preds['score'] = scores
                preds['cls_score'] = cls_scores
                preds['proposals'] = (scores_batch_idxs, proposals_idx, proposals_offset, mask_scores)

        return preds
        
    def model_fn(batch, model, epoch, semantic_only=False):
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), long, cuda
        instance_cls = batch['instance_cls'].cuda()            # (total_nInst), int, cuda
        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda
        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        voxel_feats = softgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, 'train', semantic_only=semantic_only)
        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
        
        if(epoch > cfg.prepare_epochs) and not semantic_only:
            scores_batch_idxs, cls_scores, scores, proposals_idx, proposals_offset, mask_scores = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # mask_scores: (sumNPoint, 1), float, cuda

        loss_inp = {}

        loss_inp['semantic_scores'] = (semantic_scores, labels)
        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)

        if(epoch > cfg.prepare_epochs) and not semantic_only:
            loss_inp['proposal_scores'] = (scores_batch_idxs, cls_scores, scores, proposals_idx, proposals_offset, instance_pointnum, instance_cls, mask_scores)

        loss, loss_out = loss_fn(loss_inp, epoch, semantic_only=semantic_only)

        # accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if(epoch > cfg.prepare_epochs) and not semantic_only:
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch, semantic_only=False):

        loss_out = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        
        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)

        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long


        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)       

        valid = (instance_labels != cfg.ignore_label).float()

        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())

        if (epoch > cfg.prepare_epochs) and not semantic_only:
            '''score and mask loss'''
            
            scores_batch_idxs, cls_scores, scores, proposals_idx, proposals_offset, instance_pointnum, instance_cls, mask_scores = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            # prepare to compute iou and mask target
            proposals_idx = proposals_idx[:, 1].cuda()
            proposals_offset = proposals_offset.cuda()

            # get iou and calculate mask label and mask loss
            # mask_scores_sigmoid = torch.sigmoid(mask_scores)

            # if getattr(cfg, 'cal_iou_based_on_mask', False) \
            #         and (epoch > cfg.cal_iou_based_on_mask_start_epoch):
            #     ious, mask_label =  softgroup_ops.cal_iou_and_masklabel(proposals_idx[:, 1].cuda(), \
            #         proposals_offset.cuda(), instance_labels, instance_cls, instance_pointnum, mask_scores_sigmoid.detach(), 1)
            # else:
            #     ious, mask_label =  softgroup_ops.cal_iou_and_masklabel(proposals_idx[:, 1].cuda(), \
            #         proposals_offset.cuda(), instance_labels, instance_cls, instance_pointnum, mask_scores_sigmoid.detach(), 0)

            # cal iou of clustered instance
            ious_on_cluster = softgroup_ops.get_mask_iou_on_cluster(proposals_idx,
                    proposals_offset, instance_labels, instance_pointnum)

           
            # filter out stuff instance
            fg_inds = (instance_cls != cfg.ignore_label)
            fg_instance_cls = instance_cls[fg_inds]
            fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

            # overlap > thr on fg instances are positive samples
            max_iou, gt_inds = fg_ious_on_cluster.max(1)
            pos_inds = max_iou >= cfg.iou_thr  # this value should match thr in get_mask_label.cu
            pos_gt_inds = gt_inds[pos_inds]

            # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
            labels = fg_instance_cls.new_full((fg_ious_on_cluster.size(0), ), cfg.classes)
            labels[pos_inds] = fg_instance_cls[pos_gt_inds]
            cls_loss = F.cross_entropy(cls_scores, labels)
            loss_out['cls_loss'] = (cls_loss, labels.size(0))
           
            # compute mask loss
            mask_cls_label = labels[scores_batch_idxs.long()]
            slice_inds = torch.arange(0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
            mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
            # if getattr(cfg, 'cal_iou_based_on_mask', False) \
            #         and (epoch > cfg.cal_iou_based_on_mask_start_epoch):
            #     ious =  softgroup_ops.get_mask_iou_on_pred(proposals_idx, 
            #         proposals_offset, instance_labels, instance_pointnum, mask_scores_sigmoid_slice.detach())
            # else:
            #     ious = ious_on_cluster
            mask_label = softgroup_ops.get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls, instance_pointnum, ious_on_cluster, cfg.iou_thr)
            mask_label_weight = (mask_label != -1).float()
            mask_label[mask_label==-1.] = 0.5 # any value is ok
            mask_loss = F.binary_cross_entropy(mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
            mask_loss /= (mask_label_weight.sum() + 1)
            loss_out['mask_loss'] = (mask_loss, mask_label_weight.sum())
            
            # mask_loss = torch.nn.functional.binary_cross_entropy(mask_scores_sigmoid, mask_label, weight=mask_label_weight, reduction='none')
            # mask_loss = mask_loss.mean()
            # loss_out['mask_loss'] = (mask_loss, mask_label_weight.sum())

            # compute mask score loss
            ious =  softgroup_ops.get_mask_iou_on_pred(proposals_idx, 
                proposals_offset, instance_labels, instance_pointnum, mask_scores_sigmoid_slice.detach())
            fg_ious = ious[:, fg_inds]
            gt_ious, _ = fg_ious.max(1)  # gt_ious: (nProposal) float, long
            

            # gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
            score_weight = (labels < cfg.classes).float()
            score_slice = scores[slice_inds, labels]
            score_loss = F.mse_loss(score_slice, gt_ious, reduction='none')
            score_loss = (score_loss * score_weight).sum() / (score_weight.sum() + 1)


            loss_out['score_loss'] = (score_loss, score_weight.sum())

        '''total loss'''
        loss = cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_norm_loss
        if(epoch > cfg.prepare_epochs) and not semantic_only:
            loss += (cfg.loss_weight[2] * cls_loss)
            loss += (cfg.loss_weight[3] * mask_loss)
            loss += (cfg.loss_weight[4] * score_loss)

        return loss, loss_out

    if test:
        fn = test_model_fn
    else:
        fn = model_fn

    return fn
