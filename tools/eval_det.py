# Modified from https://github.com/facebookresearch/votenet/blob/main/utils/eval_det.py
import glob
import os.path as osp
from multiprocessing import Pool

import numpy as np
import torch


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_iou(box_a, box_b, eps=1e-10):
    """Computes IoU of two axis aligned bboxes.

    Args:
        box_a, box_b: xyzxyz
    Returns:
        iou
    """

    max_a = box_a[3:]
    max_b = box_b[3:]
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3]
    min_b = box_b[0:3]
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = (box_a[3:6] - box_a[:3]).prod()
    vol_b = (box_b[3:6] - box_b[:3]).prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for a
    single class.

    Input:
        pred: map of {img_id: [(sphere, score)]} where sphere is numpy array
        gt: map of {img_id: [sphere]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'sphere': sphere list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        sphere = np.array(gt[img_id])
        det = [False] * len(sphere)
        npos += len(sphere)
        class_recs[img_id] = {'sphere': sphere, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'sphere': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for sphere, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(sphere)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R['sphere'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)


def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for
    multiple classes.

    Input:
        pred_all: map of {img_id: [(classname, sphere, score)]}
        gt_all: map of {img_id: [(classname, sphere)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, sphere, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((sphere, score))
    for img_id in gt_all.keys():
        for classname, sphere in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(sphere)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname],
                                                                      gt[classname], ovthresh,
                                                                      use_07_metric, get_iou_func)

    return rec, prec, ap


def eval_sphere(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for
    multiple classes.

    Input:
        pred_all: map of {img_id: [(classname, sphere, score)]}
        gt_all: map of {img_id: [(classname, sphere)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, sphere, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((sphere, score))
    for img_id in gt_all.keys():
        for classname, sphere in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(sphere)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_det_cls_wrapper,
                       [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
                        for classname in gt.keys() if classname in pred])
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0

    return rec, prec, ap


if __name__ == '__main__':
    CLASS_LABELS = [
        'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
        'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
        'otherfurniture'
    ]
    VALID_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    data_path = './dataset/scannetv2/val/'
    results_path = './results'
    iou_threshold = 0.25  # adjust threshold here
    instance_paths = glob.glob(osp.join(results_path, 'pred_instance', '*.txt'))
    instance_paths.sort()

    def single_process(instance_path):
        img_id = osp.basename(instance_path)[:-4]
        print('Processing', img_id)
        gt = osp.join(data_path, img_id + '_inst_nostuff.pth')  # 0-based index
        assert osp.isfile(gt)
        coords, rgb, semantic_label, instance_label = torch.load(gt)
        pred_infos = open(instance_path, 'r').readlines()
        pred_infos = [x.rstrip().split() for x in pred_infos]  # nyu_id index
        mask_path, labels, scores = list(zip(*pred_infos))
        pred = []
        for mask_path, label, score in pred_infos:
            mask_full_path = osp.join(results_path, 'pred_instance', mask_path)
            mask = np.array(open(mask_full_path).read().splitlines(), dtype=int).astype(bool)
            instance = coords[mask]
            box_min = instance.min(0)
            box_max = instance.max(0)
            box = np.concatenate([box_min, box_max])
            class_name = CLASS_LABELS[VALID_CLASS_IDS.index(int(label))]
            pred.append((class_name, box, float(score)))

        instance_num = int(instance_label.max()) + 1
        gt = []
        for i in range(instance_num):
            inds = instance_label == i
            gt_label_loc = np.nonzero(inds)[0][0]
            cls_id = int(semantic_label[gt_label_loc])
            if cls_id >= 2:
                instance = coords[inds]
                box_min = instance.min(0)
                box_max = instance.max(0)
                box = np.concatenate([box_min, box_max])
                class_name = CLASS_LABELS[cls_id - 2]
                gt.append((class_name, box))
        return img_id, pred, gt

    pool = Pool()
    results = pool.map(single_process, instance_paths)
    pool.close()
    pool.join()

    pred_all = {}
    gt_all = {}
    for img_id, pred, gt in results:
        pred_all[img_id] = pred
        gt_all[img_id] = gt

    print('Evaluating...')
    eval_res = eval_sphere(pred_all, gt_all, ovthresh=iou_threshold)
    aps = list(eval_res[-1].values())
    mAP = np.mean(aps)
    print('mAP:', mAP)
