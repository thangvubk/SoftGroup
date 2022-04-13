# Adapted from https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py  # noqa E501
# Modified by Thang Vu

import multiprocessing as mp
from copy import deepcopy

import numpy as np

from ..util import rle_decode
from .instance_eval_util import get_instances


class ScanNetEval(object):

    def __init__(self, class_labels, iou_type=None, use_label=True):
        self.valid_class_labels = class_labels
        self.valid_class_ids = np.arange(len(class_labels)) + 1
        self.id2label = {}
        self.label2id = {}
        for i in range(len(self.valid_class_ids)):
            self.label2id[self.valid_class_labels[i]] = self.valid_class_ids[i]
            self.id2label[self.valid_class_ids[i]] = self.valid_class_labels[i]

        self.ious = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = np.array([100])
        self.distance_threshes = np.array([float('inf')])
        self.distance_confs = np.array([-float('inf')])

        self.iou_type = iou_type
        self.use_label = use_label
        if self.use_label:
            self.eval_class_labels = self.valid_class_labels
        else:
            self.eval_class_labels = ['class_agnostic']

    def evaluate_matches(self, matches):
        ious = self.ious
        min_region_sizes = [self.min_region_sizes[0]]
        dist_threshes = [self.distance_threshes[0]]
        dist_confs = [self.distance_confs[0]]

        # results: class x iou
        ap = np.zeros((len(dist_threshes), len(self.eval_class_labels), len(ious)), np.float)
        rc = np.zeros((len(dist_threshes), len(self.eval_class_labels), len(ious)), np.float)
        for di, (min_region_size, distance_thresh,
                 distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
            for oi, iou_th in enumerate(ious):
                pred_visited = {}
                for m in matches:
                    for p in matches[m]['pred']:
                        for label_name in self.eval_class_labels:
                            for p in matches[m]['pred'][label_name]:
                                if 'filename' in p:
                                    pred_visited[p['filename']] = False
                for li, label_name in enumerate(self.eval_class_labels):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in matches:
                        pred_instances = matches[m]['pred'][label_name]
                        gt_instances = matches[m]['gt'][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt for gt in gt_instances
                            if gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and
                            gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float('inf'))
                        cur_match = np.zeros(len(gt_instances), dtype=np.bool)
                        # collect matches
                        for (gti, gt) in enumerate(gt_instances):
                            found_match = False
                            for pred in gt['matched_pred']:
                                # greedy assignments
                                if pred_visited[pred['filename']]:
                                    continue
                                # TODO change to use compact iou
                                iou = pred['iou']
                                if iou > iou_th:
                                    confidence = pred['confidence']
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is
                                    # automatically a FP
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred['filename']] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match == True]  # noqa E712
                        cur_score = cur_score[cur_match == True]  # noqa E712

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred['matched_gt']:
                                iou = gt['iou']
                                if iou > iou_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred['void_intersection']
                                for gt in pred['matched_gt']:
                                    # group?
                                    if gt['instance_id'] < 1000:
                                        num_ignore += gt['intersection']
                                    # small ground truth instances
                                    if (gt['vert_count'] < min_region_size
                                            or gt['med_dist'] > distance_thresh
                                            or gt['dist_conf'] < distance_conf):
                                        num_ignore += gt['intersection']
                                proportion_ignore = float(num_ignore) / pred['vert_count']
                                # if not ignored append false positive
                                if proportion_ignore <= iou_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred['confidence']
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        num_true_examples = y_true_sorted_cumsum[-1]
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # recall is the first point on recall curve
                        rc_current = recall[0]

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall[-1] = 0.

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                        rc_current = 0.0
                    else:
                        ap_current = float('nan')
                        rc_current = float('nan')
                    ap[di, li, oi] = ap_current
                    rc[di, li, oi] = rc_current
        return ap, rc

    def compute_averages(self, aps, rcs):
        d_inf = 0
        o50 = np.where(np.isclose(self.ious, 0.5))
        o25 = np.where(np.isclose(self.ious, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.ious, 0.25)))
        avg_dict = {}
        # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
        avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])
        avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
        avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
        avg_dict['all_rc'] = np.nanmean(rcs[d_inf, :, oAllBut25])
        avg_dict['all_rc_50%'] = np.nanmean(rcs[d_inf, :, o50])
        avg_dict['all_rc_25%'] = np.nanmean(rcs[d_inf, :, o25])
        avg_dict['classes'] = {}
        for (li, label_name) in enumerate(self.eval_class_labels):
            avg_dict['classes'][label_name] = {}
            avg_dict['classes'][label_name]['ap'] = np.average(aps[d_inf, li, oAllBut25])
            avg_dict['classes'][label_name]['ap50%'] = np.average(aps[d_inf, li, o50])
            avg_dict['classes'][label_name]['ap25%'] = np.average(aps[d_inf, li, o25])
            avg_dict['classes'][label_name]['rc'] = np.average(rcs[d_inf, li, oAllBut25])
            avg_dict['classes'][label_name]['rc50%'] = np.average(rcs[d_inf, li, o50])
            avg_dict['classes'][label_name]['rc25%'] = np.average(rcs[d_inf, li, o25])
        return avg_dict

    def assign_instances_for_scan(self, preds, gts):
        """get gt instances, only consider the valid class labels even in class
        agnostic setting."""
        gt_instances = get_instances(gts, self.valid_class_ids, self.valid_class_labels,
                                     self.id2label)
        # associate
        if self.use_label:
            gt2pred = deepcopy(gt_instances)
            for label in gt2pred:
                for gt in gt2pred[label]:
                    gt['matched_pred'] = []

        else:
            gt2pred = {}
            agnostic_instances = []
            # concat all the instances label to agnostic label
            for _, instances in gt_instances.items():
                agnostic_instances += deepcopy(instances)
            for gt in agnostic_instances:
                gt['matched_pred'] = []
            gt2pred[self.eval_class_labels[0]] = agnostic_instances

        pred2gt = {}
        for label in self.eval_class_labels:
            pred2gt[label] = []
        num_pred_instances = 0
        # mask of void labels in the groundtruth
        bool_void = np.logical_not(np.in1d(gts // 1000, self.valid_class_ids))
        # go thru all prediction masks
        for pred in preds:
            if self.use_label:
                label_id = pred['label_id']
                if label_id not in self.id2label:
                    continue
                label_name = self.id2label[label_id]
            else:
                label_name = self.eval_class_labels[0]  # class agnostic label
            conf = pred['conf']
            pred_mask = pred['pred_mask']
            # pred_mask can be np.array or rle dict
            if isinstance(pred_mask, dict):
                pred_mask = rle_decode(pred_mask)
            assert pred_mask.shape[0] == gts.shape[0]

            # convert to binary
            pred_mask = np.not_equal(pred_mask, 0)
            num = np.count_nonzero(pred_mask)
            if num < self.min_region_sizes[0]:
                continue  # skip if empty

            pred_instance = {}
            pred_instance['filename'] = '{}_{}'.format(pred['scan_id'], num_pred_instances)  # dummy
            pred_instance['pred_id'] = num_pred_instances
            pred_instance['label_id'] = label_id if self.use_label else None
            pred_instance['vert_count'] = num
            pred_instance['confidence'] = conf
            pred_instance['void_intersection'] = np.count_nonzero(
                np.logical_and(bool_void, pred_mask))

            # matched gt instances
            matched_gt = []
            # go thru all gt instances with matching label
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(
                    np.logical_and(gts == gt_inst['instance_id'], pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy['intersection'] = intersection
                    pred_copy['intersection'] = intersection
                    iou = (
                        float(intersection) /
                        (gt_copy['vert_count'] + pred_copy['vert_count'] - intersection))
                    gt_copy['iou'] = iou
                    pred_copy['iou'] = iou
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
            pred_instance['matched_gt'] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt

    def print_results(self, avgs):
        sep = ''
        col1 = ':'
        lineLen = 64

        print()
        print('#' * lineLen)
        line = ''
        line += '{:<15}'.format('what') + sep + col1
        line += '{:>8}'.format('AP') + sep
        line += '{:>8}'.format('AP_50%') + sep
        line += '{:>8}'.format('AP_25%') + sep
        line += '{:>8}'.format('AR') + sep
        line += '{:>8}'.format('RC_50%') + sep
        line += '{:>8}'.format('RC_25%') + sep

        print(line)
        print('#' * lineLen)

        for (li, label_name) in enumerate(self.eval_class_labels):
            ap_avg = avgs['classes'][label_name]['ap']
            ap_50o = avgs['classes'][label_name]['ap50%']
            ap_25o = avgs['classes'][label_name]['ap25%']
            rc_avg = avgs['classes'][label_name]['rc']
            rc_50o = avgs['classes'][label_name]['rc50%']
            rc_25o = avgs['classes'][label_name]['rc25%']
            line = '{:<15}'.format(label_name) + sep + col1
            line += sep + '{:>8.3f}'.format(ap_avg) + sep
            line += sep + '{:>8.3f}'.format(ap_50o) + sep
            line += sep + '{:>8.3f}'.format(ap_25o) + sep
            line += sep + '{:>8.3f}'.format(rc_avg) + sep
            line += sep + '{:>8.3f}'.format(rc_50o) + sep
            line += sep + '{:>8.3f}'.format(rc_25o) + sep
            print(line)

        all_ap_avg = avgs['all_ap']
        all_ap_50o = avgs['all_ap_50%']
        all_ap_25o = avgs['all_ap_25%']
        all_rc_avg = avgs['all_rc']
        all_rc_50o = avgs['all_rc_50%']
        all_rc_25o = avgs['all_rc_25%']

        print('-' * lineLen)
        line = '{:<15}'.format('average') + sep + col1
        line += '{:>8.3f}'.format(all_ap_avg) + sep
        line += '{:>8.3f}'.format(all_ap_50o) + sep
        line += '{:>8.3f}'.format(all_ap_25o) + sep
        line += '{:>8.3f}'.format(all_rc_avg) + sep
        line += '{:>8.3f}'.format(all_rc_50o) + sep
        line += '{:>8.3f}'.format(all_rc_25o) + sep
        print(line)
        print('#' * lineLen)
        print()

    def write_result_file(self, avgs, filename):
        _SPLITTER = ','
        with open(filename, 'w') as f:
            f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
            for class_name in self.eval_class_labels:
                ap = avgs['classes'][class_name]['ap']
                ap50 = avgs['classes'][class_name]['ap50%']
                ap25 = avgs['classes'][class_name]['ap25%']
                f.write(_SPLITTER.join([str(x) for x in [class_name, ap, ap50, ap25]]) + '\n')

    def evaluate(self, pred_list, gt_list):
        """
        Args:
            pred_list:
                for each scan:
                    for each instance
                        instance = dict(scan_id, label_id, mask, conf)
            gt_list:
                for each scan:
                    for each point:
                        gt_id = class_id * 1000 + instance_id
        """
        pool = mp.Pool()
        results = pool.starmap(self.assign_instances_for_scan, zip(pred_list, gt_list))
        pool.close()
        pool.join()

        matches = {}
        for i, (gt2pred, pred2gt) in enumerate(results):
            matches_key = f'gt_{i}'
            matches[matches_key] = {}
            matches[matches_key]['gt'] = gt2pred
            matches[matches_key]['pred'] = pred2gt
        ap_scores, rc_scores = self.evaluate_matches(matches)
        avgs = self.compute_averages(ap_scores, rc_scores)

        # print
        self.print_results(avgs)
        return avgs
