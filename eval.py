#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map


'''
result = 'result.pkl'
config = 'configs/custom/faster_rcnn_r101_fpn_dcn.py'
iou_thr = 0.5
'''

def voc_eval(result_file, dataset, iou_thr=0.5):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []

    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        #if 'bboxes_ignore' in ann:
            #ignore = np.concatenate([
            #    np.zeros(bboxes.shape[0], dtype=np.bool),
            #    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            #])
            #gt_ignore.append(ignore)
            #bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            #print(ann)
            #labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)

    if not gt_ignore:
        gt_ignore = None

    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset.CLASSES,
        print_summary=True)

def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    voc_eval(args.result, test_dataset, args.iou_thr)


if __name__ == '__main__':
    main()



