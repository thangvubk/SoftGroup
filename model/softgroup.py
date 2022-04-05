import functools
import spconv
import sys
import torch
import torch.nn as nn

from util import utils
from .blocks import ResidualBlock, UBlock

sys.path.append('../../')

from lib.softgroup_ops.functions import softgroup_ops  # noqa


class SoftGroup(nn.Module):

    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 sem2ins_classes=[],
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
        self.sem2ins_classes = sem2ins_classes
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
            spconv.SubMConv3d(
                6, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # semantic segmentation branch
        self.semantic_linear = nn.Sequential(
            nn.Linear(channels, channels, bias=True), norm_fn(channels), nn.ReLU(),
            nn.Linear(channels, semantic_classes))

        # center shift vector branch
        self.offset_linear = nn.Sequential(
            nn.Linear(channels, channels, bias=True), norm_fn(channels), nn.ReLU(),
            nn.Linear(channels, 3, bias=True))

        # topdown refinement path
        self.intra_ins_unet = UBlock([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
        self.intra_ins_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
        self.cls_linear = nn.Linear(channels, instance_classes + 1)
        self.mask_linear = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, instance_classes + 1))
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
        coords = batch['locs'].cuda()
        voxel_coords = batch['voxel_locs'].cuda()
        p2v_map = batch['p2v_map'].cuda()
        v2p_map = batch['v2p_map'].cuda()
        coords_float = batch['locs_float'].cuda()
        feats = batch['feats'].cuda()
        labels = batch['labels'].cuda()
        instance_labels = batch['instance_labels'].cuda()
        # instance_info = batch['instance_info'].cuda()
        # instance_pointnum = batch['instance_pointnum'].cuda()
        # instance_cls = batch['instance_cls'].cuda()
        # batch_offsets = batch['offsets'].cuda()
        spatial_shape = batch['spatial_shape']

        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = softgroup_ops.voxelization(feats, v2p_map)
        if self.test_cfg.x4_split:
            input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, 4)
            batch_idxs = torch.zeros_like(coords[:, 0].int())
        else:
            input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, 1)
            batch_idxs = coords[:, 0].int()
        semantic_scores, pt_offsets, output_feats, coords_float = self.forward_backbone(
            input, p2v_map, coords_float, x4_split=self.test_cfg.x4_split)  # TODO check name for map
        proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                batch_idxs, coords_float,
                                                                self.grouping_cfg)
        scores_batch_idxs, cls_scores, iou_scores, mask_scores = self.forward_instance(
            proposals_idx, proposals_offset, output_feats, coords_float)
        pred_instances = self.get_instances(batch['scan_ids'][0], proposals_idx, semantic_scores,
                                            cls_scores, iou_scores, mask_scores)
        gt_instances = self.get_gt_instances(labels, instance_labels)
        ret = {}
        ret['det_ins'] = pred_instances
        ret['gt_ins'] = gt_instances
        return ret

    def forward_backbone(self, input, input_map, coords, x4_split=False):
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
        pt_offsets = self.offset_linear(output_feats)
        return semantic_scores, pt_offsets, output_feats, coords

    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: devide and forward 4 parts of a scene"""
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]

    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them"""
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

    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None):
        proposals_idx_list = []
        proposals_offset_list = []
        batch_size = batch_idxs.max() + 1
        semantic_preds = semantic_scores.max(1)[1]  # TODO remove this

        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
        training_mode = None  # TODO remove this
        for class_id in range(self.semantic_classes):
            # ignore "floor" and "wall"
            if class_id < 2:
                continue
            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            if object_idxs.size(0) < 100:  # TODO
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]  # (N_fg, 3), float32
            semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()
            idx, start_len = softgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                             batch_offsets_, radius, mean_active)
            using_set_aggr = False  # TODO refactor this
            proposals_idx, proposals_offset = softgroup_ops.hierarchical_aggregation(
                class_numpoint_mean, semantic_preds_cpu, (coords_ + pt_offsets_).cpu(), idx.cpu(),
                start_len.cpu(), batch_idxs_.cpu(), training_mode, using_set_aggr, class_id)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

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
        input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                          output_feats, coords_float,
                                                          **self.instance_voxel_cfg)

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

    def get_instances(self, scan_id, proposals_idx, semantic_scores, cls_scores, iou_scores,
                      mask_scores):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            if i in self.sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
            else:
                cls_pred = cls_scores.new_full((num_instances, ), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]
                cur_iou_scores = iou_scores[:, i]
                cur_mask_scores = mask_scores[:, i]
                score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
                mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')
                mask_inds = cur_mask_scores > self.test_cfg.mask_score_thr
                cur_proposals_idx = proposals_idx[mask_inds].long()
                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

                # filter low score instance
                inds = cur_cls_scores > self.test_cfg.cls_score_thr
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]

                # filter too small instances
                npoint = mask_pred.sum(1)
                inds = npoint >= self.test_cfg.min_npoint
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)
        cls_pred = torch.cat(cls_pred_list).cpu().numpy()
        score_pred = torch.cat(score_pred_list).cpu().numpy()
        mask_pred = torch.cat(mask_pred_list).cpu().numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            pred['pred_mask'] = mask_pred[i]
            instances.append(pred)
        return instances

    def get_gt_instances(self, labels, instance_labels):
        """Get gt instances for evaluation"""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        labels = labels - label_shift + 1
        labels[labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        gt_ins = labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, scale,
                              spatial_shape):
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = softgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0,
                                                  clusters_idx[:, 0].cuda().long())
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = softgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())
        clusters_coords_max = softgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())

        clusters_scale = 1 / (
            (clusters_coords_max - clusters_coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = -min_xyz + torch.clamp(
            spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(
                spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) *
                                                 (clusters_coords < spatial_shape)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(),
                                     clusters_coords.cpu()], 1)

        out_coords, inp_map, out_map = softgroup_ops.voxelization_idx(clusters_coords,
                                                                      int(clusters_idx[-1, 0]) + 1)
        out_feats = softgroup_ops.voxelization(clusters_feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
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
