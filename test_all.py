import torch
import time
import numpy as np
import random
import os

from util.config import cfg
cfg.task = 'test'
from util.log import logger
import util.utils as utils
import util.eval as eval
import glob

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            from data.scannetv2_inst import Dataset
            dataset = Dataset(test=True)
            dataset.testLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
    dataloader = dataset.test_data_loader
    total = 0

    with torch.no_grad():
        model = model.eval()

        total_end1 = 0.
        matches = {}
        for i, batch in enumerate(dataloader):

            # inference
            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1

            # decode results for evaluation
            N = batch['feats'].shape[0]
            test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:12]
            semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
            pt_offsets = preds['pt_offsets']    # (N, 3), float32, cuda
            if (epoch > cfg.prepare_epochs):
                scores = preds['score']   # (nProposal, 1) float, cuda
                # scores_pred = torch.sigmoid(scores.view(-1))

                scores_batch_idxs, proposals_idx, proposals_offset, mask_scores = preds['proposals']
                cls_scores = preds['cls_score'].softmax(1)
                slice_inds = torch.arange(cls_scores.size(0), dtype=torch.long, device=cls_scores.device)
                cls_scores_new, cls_pred = cls_scores[:, :-1].max(1)

                cluster_scores_list = []
                clusters_list = []
                cluster_semantic_id_list = []
                # import pdb; pdb.set_trace()
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
                    test_mask_score_thre = getattr(cfg, 'test_mask_score_thre', -0.5)
                    _mask = mask_scores_new > test_mask_score_thre
                    proposals_pred[proposals_idx[_mask][:, 0].long(), proposals_idx[_mask][:, 1].long()] = 1

                    # bg filtering
                    # import pdb; pdb.set_trace()
                    # pos_inds = (cls_pred != cfg.classes - 2)
                    # proposals_pred = proposals_pred[pos_inds]
                    # scores_pred = scores_pred[pos_inds]
                    # cls_pred = cls_pred[pos_inds]
                    # import pdb; pdb.set_trace()
                    semantic_id = cls_scores.new_full(cls_scores_new.size(), semantic_label_idx[i + 2], dtype=torch.long)


                    semantic_id1 = torch.tensor(semantic_label_idx, device=scores_pred.device) \
                        [semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long
                    # semantic_id_idx = semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]
    
                    proposals_pred = proposals_pred[score_inds]
                    scores_pred = scores_pred[score_inds]
                    semantic_id = semantic_id[score_inds]

                    # score threshold
                    score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                    scores_pred = scores_pred[score_mask]
                    proposals_pred = proposals_pred[score_mask]
                    semantic_id = semantic_id[score_mask]
                    # semantic_id_idx = semantic_id_idx[score_mask]

                    # npoint threshold
                    proposals_pointnum = proposals_pred.sum(1)
                    npoint_mask = (proposals_pointnum >= cfg.TEST_NPOINT_THRESH)
                    scores_pred = scores_pred[npoint_mask]
                    proposals_pred = proposals_pred[npoint_mask]
                    semantic_id = semantic_id[npoint_mask]


                    # nms (no need)
                    if getattr(cfg, 'using_NMS', False):
                        if semantic_id.shape[0] == 0:
                            pick_idxs = np.empty(0)
                        else:
                            proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                            intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                            proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                            proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                            proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                            cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                            pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), cfg.TEST_NMS_THRESH)  
                            # int, (nCluster, N)
                        clusters = proposals_pred[pick_idxs]
                        cluster_scores = scores_pred[pick_idxs]
                        cluster_semantic_id = semantic_id[pick_idxs]
                    else:
                        clusters = proposals_pred
                        cluster_scores = scores_pred
                        cluster_semantic_id = semantic_id
                    clusters_list.append(clusters)
                    cluster_scores_list.append(cluster_scores)
                    cluster_semantic_id_list.append(cluster_semantic_id)
                clusters = torch.cat(clusters_list)
                cluster_scores = torch.cat(cluster_scores_list)
                cluster_semantic_id = torch.cat(cluster_semantic_id_list)

                nclusters = clusters.shape[0]
                if nclusters > cfg.max_clusters:
                    nclusters = cfg.max_clusters
                    _, topk_inds = cluster_scores.topk(cfg.max_clusters)
                    clusters = clusters[topk_inds]
                    cluster_scores = cluster_scores[topk_inds]
                    cluster_semantic_id = cluster_semantic_id[topk_inds]


                # prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy()
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                    pred_info['mask'] = clusters.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)

                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt
                
                    if cfg.split == 'val':
                        matches[test_scene_name]['seg_gt'] = batch['labels']
                        matches[test_scene_name]['seg_pred'] = semantic_pred
                # break
    

            # save files
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch['locs_float'].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)

            if(epoch > cfg.prepare_epochs and cfg.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format( \
                        test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')
                    np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                f.close()


            logger.info("instance iter: {}/{} point_num: {} ncluster: {} inference time: {:.2f}s".format( \
                batch['id'][0] + 1, len(dataset.test_files), N, nclusters, end1))
            total_end1 += end1
            # import pdb; pdb.set_trace()
            # break

        # evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs)

        logger.info("whole set inference time: {:.2f}s, latency per frame: {:.2f}ms".format(total_end1, total_end1 / len(dataloader) * 1000))

        # evaluate semantic segmantation accuracy and mIoU
        if cfg.split == 'val':
            seg_accuracy = evaluate_semantic_segmantation_accuracy(matches)
            logger.info("semantic_segmantation_accuracy: {:.4f}".format(seg_accuracy))
            miou = evaluate_semantic_segmantation_miou(matches)
            logger.info("semantic_segmantation_mIoU: {:.4f}".format(miou))
        return avgs

def evaluate_semantic_segmantation_accuracy(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    correct = (seg_gt_all[seg_gt_all != -100] == seg_pred_all[seg_gt_all != -100]).sum()
    whole = (seg_gt_all != -100).sum()
    seg_accuracy = correct.float() / whole.float()
    return seg_accuracy

def evaluate_semantic_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    iou_list = []
    for _index in seg_gt_all.unique():
        if _index != -100:
            intersection = ((seg_gt_all == _index) &  (seg_pred_all == _index)).sum()
            union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
            iou = intersection.float() / union
            iou_list.append(iou)
    iou_tensor = torch.tensor(iou_list)
    miou = iou_tensor.mean()
    return miou

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    init()

    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if model_name == 'hais':
        from model.hais.hais import HAIS as Network
        from model.hais.hais import model_fn_decorator
    
    else:
        print("Error: no model version " + model_name)
        exit(0)
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))
    model_fn = model_fn_decorator(test=True)

    pretrain_models = glob.glob(cfg.exp_path + '/*.pth')
    pretrain_models = sorted(pretrain_models)
    epochs = [p[-7:-4] for p in pretrain_models]
    for epoch, pretrain in zip(epochs, pretrain_models):
        if int(epoch) < 400:
            continue
        print(pretrain)
        # load model
        utils.checkpoint_restore(cfg, model, None, cfg.exp_path, cfg.config.split('/')[-1][:-5], 
            use_cuda, cfg.test_epoch, dist=False, f=pretrain)      
        # resume from the latest epoch, or specify the epoch to restore

        # evaluate
        avgs = test(model, model_fn, data_name, cfg.test_epoch)
        os.makedirs(os.path.join(cfg.exp_path, 'test_log'), exist_ok=True)
        test_log = os.path.join(cfg.exp_path, 'test_log', 'test_log.csv')
        with_header = not os.path.exists(test_log)
        with open(test_log, 'a') as f:
            if with_header:
                header = 'Epoch,AP,AP50,AP25,'
            line = '{},{:.3f},{:.3f},{:.3f},'.format(epoch, avgs['all_ap'], avgs['all_ap_50%'], avgs['all_ap_25%'])
            for class_name, aps in avgs['classes'].items():
                if with_header:
                    header += 'AP_{},'.format(class_name)
                line += '{:.3f},'.format(aps['ap'])
            for class_name, aps in avgs['classes'].items():
                if with_header:
                    header += 'AP50_{},'.format(class_name)
                line += '{:.3f},'.format(aps['ap50%'])
            for class_name, aps in avgs['classes'].items():
                if with_header:
                    header += 'AP25_{},'.format(class_name)
                line += '{:.3f},'.format(aps['ap25%'])
            line += '\n'
            if with_header:
                header += '\n'
                f.write(header)
            f.write(line)
