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


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),                                   
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),                                             
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            output.features = torch.cat((identity.features, output_decoder.features), dim=1)
            output = self.blocks_tail(output)
        return output

class SoftGroup(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        input_c = cfg.input_channel
        width = cfg.width
        semantic_classes = cfg.semantic_classes
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.point_aggr_radius = cfg.point_aggr_radius
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.score_mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs
        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module
        

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        self.cfg = cfg

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, width, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.unet = UBlock([width, 2*width, 3*width, 4*width, 5*width, 6*width, 7*width], norm_fn, block_reps, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(
            norm_fn(width),
            nn.ReLU()
        )

        # semantic segmentation branch
        self.semantic_linear = nn.Sequential(
            nn.Linear(width, width, bias=True),
            norm_fn(width),
            nn.ReLU(),
            nn.Linear(width, semantic_classes)
        )

        # center shift vector branch
        self.offset_linear = nn.Sequential(
            nn.Linear(width, width, bias=True),
            norm_fn(width),
            nn.ReLU(),
            nn.Linear(width, 3, bias=True)
        )

        # topdown refinement path
        self.intra_ins_unet = UBlock([width, 2*width], norm_fn, 2, block, indice_key_id=11)
        self.intra_ins_outputlayer = spconv.SparseSequential(
            norm_fn(width),
            nn.ReLU()
        )
        self.cls_linear = nn.Linear(width, classes + 1)
        self.mask_linear = nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, classes + 1))
        self.score_linear = nn.Linear(width, classes + 1)

        self.apply(self.set_bn_init)
        nn.init.normal_(self.score_linear.weight, 0, 0.01)
        nn.init.constant_(self.score_linear.bias, 0)

        # fix module
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'semantic_linear': self.semantic_linear, 'offset_linear': self.offset_linear,
                      'intra_ins_unet': self.intra_ins_unet, 'intra_ins_outputlayer': self.intra_ins_outputlayer, 
                      'score_linear': self.score_linear, 'mask_linear': self.mask_linear,
                      'cls_linear': self.cls_linear}
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        # load pretrain weights
        if pretrained and self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict['net'], prefix=m))


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
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

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = softgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = softgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
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

    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch, training_mode, gt_instances=None, split=False, semantic_only=False):
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
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords, self.score_fullscale, self.score_scale, self.score_mode)

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
