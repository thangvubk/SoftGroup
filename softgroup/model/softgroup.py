import functools
from collections import OrderedDict

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_decode, rle_encode
from .blocks import MLP, ResidualBlock, UBlock


class SoftGroup(nn.Module):

    def __init__(self,
                 in_channels=3,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 semantic_weight=None,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 with_coords=True,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[]):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.with_coords = with_coords
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        if with_coords:
            in_channels += 3
            self.in_channels += 3
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)

        # topdown refinement path
        if not semantic_only:
            self.tiny_unet = UBlock([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
            self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
            self.cls_linear = nn.Linear(channels, instance_classes + 1)
            self.mask_linear = MLP(channels, instance_classes + 1, norm_fn=None, num_layers=2)
            self.iou_score_linear = nn.Linear(channels, instance_classes + 1)

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()
        if not self.semantic_only:
            for m in [self.cls_linear, self.iou_score_linear]:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, spatial_shape, batch_size, **kwargs):
        losses = {}
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map)

        # point wise losses
        point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, semantic_labels,
                                               instance_labels, pt_offset_labels)
        losses.update(point_wise_loss)

        # instance losses
        if not self.semantic_only:
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)
            if proposals_offset.shape[0] > self.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]
            inst_feats, inst_map = self.clusters_voxelization(
                proposals_idx,
                proposals_offset,
                output_feats,
                coords_float,
                rand_quantize=True,
                **self.instance_voxel_cfg)
            instance_batch_idxs, cls_scores, iou_scores, mask_scores = self.forward_instance(
                inst_feats, inst_map)
            instance_loss = self.instance_loss(cls_scores, mask_scores, iou_scores, proposals_idx,
                                               proposals_offset, instance_labels, instance_pointnum,
                                               instance_cls, instance_batch_idxs)
            losses.update(instance_loss)
        return self.parse_losses(losses)

    def point_wise_loss(self, semantic_scores, pt_offsets, semantic_labels, instance_labels,
                        pt_offset_labels):
        losses = {}
        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device='cuda')
        else:
            weight = None
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, weight=weight, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
        else:
            offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
        losses['offset_loss'] = offset_loss
        return losses

    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def instance_loss(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls, instance_batch_idxs):
        if proposals_idx.size(0) == 0 or (instance_cls != self.ignore_label).sum() == 0:
            cls_loss = cls_scores.sum() * 0
            mask_loss = mask_scores.sum() * 0
            iou_score_loss = iou_scores.sum() * 0
            return dict(
                cls_loss=cls_loss,
                mask_loss=mask_loss,
                iou_score_loss=iou_score_loss,
                num_pos=mask_loss,
                num_neg=mask_loss)

        losses = {}
        proposals_idx = proposals_idx[:, 1].int().cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                                  instance_pointnum)

        # filter out background instances
        fg_inds = (instance_cls != self.ignore_label)
        fg_instance_cls = instance_cls[fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # assign proposal to gt idx. -1: negative, 0 -> num_gts - 1: positive
        num_proposals = fg_ious_on_cluster.size(0)
        num_gts = fg_ious_on_cluster.size(1)
        assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals, ), -1, dtype=torch.long)

        # overlap > thr on fg instances are positive samples
        max_iou, argmax_iou = fg_ious_on_cluster.max(1)
        pos_inds = max_iou >= self.train_cfg.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]

        # allow low-quality proposals with best iou to be as positive sample
        # in case pos_iou_thr is too high to achieve
        match_low_quality = getattr(self.train_cfg, 'match_low_quality', False)
        min_pos_thr = getattr(self.train_cfg, 'min_pos_thr', 0)
        if match_low_quality:
            gt_max_iou, gt_argmax_iou = fg_ious_on_cluster.max(0)
            for i in range(num_gts):
                if gt_max_iou[i] >= min_pos_thr:
                    assigned_gt_inds[gt_argmax_iou[i]] = i

        # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
        labels = fg_instance_cls.new_full((num_proposals, ), self.instance_classes)
        pos_inds = assigned_gt_inds >= 0
        labels[pos_inds] = fg_instance_cls[assigned_gt_inds[pos_inds]]
        cls_loss = F.cross_entropy(cls_scores, labels)
        losses['cls_loss'] = cls_loss

        # compute mask loss
        mask_cls_label = labels[instance_batch_idxs.long()]
        slice_inds = torch.arange(
            0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
        mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
        mask_label = get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                                    instance_pointnum, ious_on_cluster, self.train_cfg.pos_iou_thr)
        mask_label_weight = (mask_label != -1).float()
        mask_label[mask_label == -1.] = 0.5  # any value is ok
        mask_loss = F.binary_cross_entropy(
            mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
        mask_loss /= (mask_label_weight.sum() + 1)
        losses['mask_loss'] = mask_loss

        # compute iou score loss
        ious = get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, mask_scores_sigmoid_slice.detach())
        fg_ious = ious[:, fg_inds]
        gt_ious, _ = fg_ious.max(1)
        slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = (labels < self.instance_classes).float()
        iou_score_slice = iou_scores[slice_inds, labels]
        iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
        losses['iou_score_loss'] = iou_score_loss

        # add logging variables
        losses['num_pos'] = (labels < self.instance_classes).sum().float()
        losses['num_neg'] = (labels >= self.instance_classes).sum().float()
        return losses

    def parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' + f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    @cuda_cast
    def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):
        color_feats = feats
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        # lvl_fusion directly use output point as level 1 for pyramid map for fast inference
        lvl_fusion = getattr(self.test_cfg, 'lvl_fusion', False)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(
            input, v2p_map, x4_split=self.test_cfg.x4_split, lvl_fusion=lvl_fusion)
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
            instance_labels = self.merge_4_parts(instance_labels)
            pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(scan_id=scan_ids[0])
        if 'semantic' in self.test_cfg.eval_tasks or 'panoptic' in self.test_cfg.eval_tasks:
            ret.update(
                dict(
                    semantic_labels=semantic_labels.cpu().numpy(),
                    instance_labels=instance_labels.cpu().numpy()))
        if 'semantic' in self.test_cfg.eval_tasks:
            point_wise_results = self.get_point_wise_results(coords_float, color_feats,
                                                             semantic_preds, pt_offsets,
                                                             pt_offset_labels, v2p_map, lvl_fusion)
            ret.update(point_wise_results)
        if not self.semantic_only:
            if 'instance' in self.test_cfg.eval_tasks or 'panoptic' in self.test_cfg.eval_tasks:
                if lvl_fusion:
                    batch_idxs = input.indices[:, 0].int()
                    coords_float = voxelization(coords_float, p2v_map)
                proposals_idx, proposals_offset = self.forward_grouping(
                    semantic_scores,
                    pt_offsets,
                    batch_idxs,
                    coords_float,
                    self.grouping_cfg,
                    lvl_fusion=lvl_fusion)
                inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                                  output_feats, coords_float,
                                                                  **self.instance_voxel_cfg)
                _, cls_scores, iou_scores, mask_scores = self.forward_instance(inst_feats, inst_map)
                pred_instances = self.get_instances(
                    scan_ids[0],
                    proposals_idx,
                    semantic_scores,
                    cls_scores,
                    iou_scores,
                    mask_scores,
                    v2p_map=v2p_map,
                    lvl_fusion=lvl_fusion)
            if 'instance' in self.test_cfg.eval_tasks:
                gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
                ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))
            if 'panoptic' in self.test_cfg.eval_tasks:
                panoptic_preds = self.panoptic_fusion(semantic_preds.cpu().numpy(), pred_instances)
                ret.update(panoptic_preds=panoptic_preds)
        return ret

    def forward_backbone(self, input, input_map, x4_split=False, lvl_fusion=False):
        if x4_split:
            assert not lvl_fusion, 'x4_split not support lvl_fusion'
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features
            if not lvl_fusion:
                output_feats = output_feats[input_map.long()]

        semantic_scores = self.semantic_linear(output_feats)
        pt_offsets = self.offset_linear(output_feats)
        return semantic_scores, pt_offsets, output_feats

    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
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
        """Helper function for s3dis: take output of 4 parts and merge them."""
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

    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None,
                         lvl_fusion=False):
        proposals_idx_list = []
        proposals_offset_list = []
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        npoint_thr = self.grouping_cfg.npoint_thr
        with_pyramid = getattr(self.grouping_cfg, 'with_pyramid', False)
        with_octree = getattr(self.grouping_cfg, 'with_octree', False)
        base_size = getattr(self.grouping_cfg, 'pyramid_base_size', 0.02)
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
        assert class_numpoint_mean.size(0) == self.semantic_classes
        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            if object_idxs.size(0) < self.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            if with_pyramid:
                num_points = coords_.size(0)
                level = self.get_level(num_points)
                radius = self.grouping_cfg.radius * level
                if level > 1 or not lvl_fusion:
                    coords_, pt_offsets_, batch_idxs_, l2p_map = self.pyramid_map(
                        coords_, pt_offsets_, batch_idxs_, level, base_size)
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            neighbor_inds, start_len = ball_query(
                coords_ + pt_offsets_,
                batch_idxs_,
                batch_offsets_,
                radius,
                mean_active,
                with_octree=with_octree)
            proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, neighbor_inds.cpu(),
                                                          start_len.cpu(), npoint_thr, class_id)
            if with_pyramid:
                if level > 1 or not lvl_fusion:
                    proposals_idx, proposals_offset = self.pyramid_inverse_map(
                        proposals_idx, proposals_offset, coords_.size(0), l2p_map)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)
        if len(proposals_idx_list) > 0:
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
        else:
            proposals_idx = torch.zeros((0, 2), dtype=torch.int32)
            proposals_offset = torch.zeros((0, ), dtype=torch.int32)
        return proposals_idx, proposals_offset

    def get_level(self, num_points):
        if num_points > 1000000:
            level = 3
        elif num_points > 100000:
            level = 2
        else:
            level = 1
        return level

    def pyramid_map(self, coords_float, pt_offsets, batch_idxs, level=1, base_size=0.02):
        coords = (coords_float / (base_size * level)).long()
        coords = torch.cat([batch_idxs[:, None], coords], dim=1)
        coords, l2p_map, p2l_map = voxelization_idx(coords.cpu(), batch_idxs[-1].item() + 1)
        coords_float = voxelization(coords_float, p2l_map.cuda())
        pt_offsets = voxelization(pt_offsets, p2l_map.cuda())
        batch_idxs = coords[:, 0].cuda().int()
        return coords_float, pt_offsets, batch_idxs, l2p_map

    def pyramid_inverse_map(self, proposals_idx, proposals_offset, num_points, l2p_map):
        proposals = torch.zeros((proposals_offset.size(0) - 1, num_points), dtype=torch.int)
        proposals[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        proposals = proposals[:, l2p_map.cpu().long()]
        proposals_idx = proposals.nonzero()
        proposals_offset = torch.cumsum(proposals.sum(1), dim=0).int()
        proposals_offset = torch.cat([proposals_offset.new_zeros(1), proposals_offset])
        return proposals_idx, proposals_offset

    def forward_instance(self, inst_feats, inst_map):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores
        feats = self.global_pool(feats)
        cls_scores = self.cls_linear(feats)
        iou_scores = self.iou_score_linear(feats)
        return instance_batch_idxs, cls_scores, iou_scores, mask_scores

    @force_fp32(apply_to=('semantic_preds', 'offset_preds'))
    def get_point_wise_results(self, coords_float, color_feats, semantic_preds, offset_preds,
                               offset_labels, v2p_map, lvl_fusion):
        if lvl_fusion:
            semantic_preds = semantic_preds[v2p_map.long()]
            offset_preds = offset_preds[v2p_map.long()]
        return dict(
            coords_float=coords_float.cpu().numpy(),
            color_feats=color_feats.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            offset_preds=offset_preds.cpu().numpy(),
            offset_labels=offset_labels.cpu().numpy())

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self,
                      scan_id,
                      proposals_idx,
                      semantic_scores,
                      cls_scores,
                      iou_scores,
                      mask_scores,
                      v2p_map=None,
                      lvl_fusion=False):
        if proposals_idx.size(0) == 0:
            return []

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
                if lvl_fusion:
                    mask_pred = mask_pred[:, v2p_map.long()]
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

                if lvl_fusion:
                    mask_pred = mask_pred[:, v2p_map.long()]

                # filter too small instances
                npoint = mask_pred.sum(1)
                inds = npoint >= self.test_cfg.min_npoint
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
            cls_pred_list.append(cls_pred.cpu())
            score_pred_list.append(score_pred.cpu())
            mask_pred_list.append(mask_pred.cpu())
        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def panoptic_fusion(self, semantic_preds, instance_preds):
        cls_offset = self.semantic_classes - self.instance_classes - 1
        panoptic_cls = semantic_preds.copy().astype(np.uint32)
        panoptic_ids = np.zeros_like(semantic_preds).astype(np.uint32)

        # higher score has higher fusion priority
        scores = [x['conf'] for x in instance_preds]
        score_inds = np.argsort(scores)[::-1]
        prev_paste = np.zeros_like(semantic_preds, dtype=bool)
        panoptic_id = 1
        for i in score_inds:
            instance = instance_preds[i]
            cls = instance['label_id']
            mask = rle_decode(instance['pred_mask']).astype(bool)

            # check overlap with pasted instances
            intersect = (mask * prev_paste).sum()
            if intersect / (mask.sum() + 1e-5) > self.test_cfg.panoptic_skip_iou:
                continue

            paste = mask * (~prev_paste)
            panoptic_cls[paste] = cls + cls_offset
            panoptic_ids[paste] = panoptic_id
            prev_paste[paste] = 1
            panoptic_id += 1

        # if thing classes have panoptic id == 0, ignore it
        ignore_inds = (panoptic_cls >= 11) & (panoptic_ids == 0)

        # encode panoptic results
        panoptic_preds = (panoptic_cls & 0xFFFF) | (panoptic_ids << 16)
        panoptic_preds[ignore_inds] = self.semantic_classes
        panoptic_preds = panoptic_preds.astype(np.uint32)
        return panoptic_preds

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        if clusters_idx.size(0) == 0:
            # create dummpy tensors
            coords = torch.tensor(
                [[0, 0, 0, 0], [0, spatial_shape - 1, spatial_shape - 1, spatial_shape - 1]],
                dtype=torch.int,
                device='cuda')
            feats = feats[0:2]
            voxelization_feats = spconv.SparseConvTensor(feats, coords, [spatial_shape] * 3, 1)
            inp_map = feats.new_zeros((1, ), dtype=torch.long)
            return voxelization_feats, inp_map

        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = sec_min(coords, clusters_offset.cuda())
        coords_max = sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x
