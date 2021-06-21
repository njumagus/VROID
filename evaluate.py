import os
import json
import argparse

from collections import defaultdict


class MatrixTri():
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, matrix_tri):
        self.tp += matrix_tri.tp
        self.fp += matrix_tri.fp
        self.fn += matrix_tri.fn
        return self

    def precision(self, mAP=False):
        if mAP and self.tp + self.fn == 0:
            return -1
        if self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def Y(self, max_tp):
        if max_tp == 0:
            return 0
        return self.tp / max_tp


class Matrix():
    def __init__(self):
        self.matrix_per_tri = defaultdict(MatrixTri)

    def __getitem__(self, tri):
        return self.matrix_per_tri[tri]

    def __iadd__(self, matrix):
        for tri, matrix_tri in matrix.matrix_per_tri.items():
            self.matrix_per_tri[tri] += matrix_tri
        return self

    def metrics(self, beta2, max_tp):
        matrix_all = MatrixTri()
        mAP, n = 0, 0
        for matrix_tri in self.matrix_per_tri.values():
            matrix_all += matrix_tri
            precision = matrix_tri.precision(mAP=True)
            if precision == -1:
                continue
            n += 1
            mAP += precision

        precision = matrix_all.precision()
        recall = matrix_all.recall()
        Y = matrix_all.Y(max_tp)

        if precision == 0 and recall == 0:
            f = 0
        else:
            f = (beta2 + 1) * precision * recall / (beta2 * precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'Y': Y,
            'f': f,
            'mAP': mAP / n,
            'n': n
        }


def compute_iou(box1, box2):
    """ box: [y1, x1, y2, x2] """

    x1 = max(box1[1], box2[1])
    y1 = max(box1[0], box2[0])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[2], box2[2])

    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / float(area1 + area2 - intersection)


def compute_matrix(gt, pred, top_n, iou_thresh):
    gt_relas = gt['relation_ids']
    gt_subjs = gt['subject_class_ids']
    gt_subj_boxes = gt['subject_boxes']
    gt_objs = gt['object_class_ids']
    gt_obj_boxes = gt['object_boxes']
    num_gt = len(gt_relas)

    pred_relas = pred['relation_ids']
    pred_subjs = pred['subject_class_ids']
    pred_subj_boxes = pred['subject_boxes']
    pred_objs = pred['object_class_ids']
    pred_obj_boxes = pred['object_boxes']
    scores = pred['scores']
    num_pred = len(pred_relas)

    if top_n == 'α':
        top_n = num_gt
    indices = sorted(range(num_pred), key=lambda i: -scores[i])[:top_n]

    matrix = Matrix()
    matched_gt, matched_pred = set(), set()
    for j in indices:
        for k in range(num_gt):
            if k in matched_gt:
                continue
            if (pred_subjs[j], pred_relas[j], pred_objs[j]) != (gt_subjs[k], gt_relas[k], gt_objs[k]):
                continue
            subj_iou = compute_iou(pred_subj_boxes[j], gt_subj_boxes[k])
            obj_iou = compute_iou(pred_obj_boxes[j], gt_obj_boxes[k])
            if subj_iou > iou_thresh and obj_iou > iou_thresh:
                matched_gt.add(k)
                matched_pred.add(j)
                matrix[(gt_subjs[k], gt_relas[k], gt_objs[k])].tp += 1
                break

    for j in indices:
        if j not in matched_pred:
            matrix[(pred_subjs[j], pred_relas[j], pred_objs[j])].fp += 1

    for k in range(num_gt):
        if k not in matched_gt:
            matrix[(gt_subjs[k], gt_relas[k], gt_objs[k])].fn += 1

    max_tp = min(top_n, num_gt)

    return matrix, max_tp


def compute_metrics(gt_dict, pred_dict, top_n, beta2, iou_thresh):
    matrix, max_tp = Matrix(), 0
    num_img = 0
    for img_id in gt_dict:
        if img_id not in pred_dict:
            print('Skip {}'.format(img_id))
            continue
        img_matrix, img_max_tp = compute_matrix(gt_dict[img_id], pred_dict[img_id], top_n, iou_thresh)
        matrix += img_matrix
        max_tp += img_max_tp
        num_img += 1
    return num_img, matrix.metrics(beta2, max_tp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate precision, recall and f.')
    parser.add_argument('--gt_json', dest='gt_json', type=str, default='gt.json',
                        help='gt json file path, default: "./gt.json"')
    parser.add_argument('--pred_json', dest='pred_json', type=str, required=True,
                        help='pred json file path')
    parser.add_argument('--top_n', dest='top_n', type=int, default=0,
                        help='precision/recall @ top-n scored relations, 0 means top-α, default: 0')
    parser.add_argument('--beta2', dest='beta2', type=float, choices=[0.3, 1], default=0.3,
                        help='beta2 of F-score, 0.3 or 1, default: 0.3')
    parser.add_argument('--iou_thresh', dest='iou_thresh', type=float, default=0.5,
                        help='bounding box IOU threshold, default: 0.5')
    args = parser.parse_args()

    top_n = args.top_n if args.top_n != 0 else 'α'
    print('-- Config --')
    # print('gt json:', args.gt_json)
    print('prediction json:', args.pred_json)
    print('precision/recall @ top-n:', top_n)
    # print('F-score beta2:', args.beta2)
    # print('IOU threshold:', args.iou_thresh)
    print()
    gt_dict={}
    if not os.path.exists(args.gt_json):
        raw_gt_instance_dict=json.load(open("../data/viroi_json/test_images_dict.json"))
        raw_gt_triplet_dict=json.load(open("../data/viroi_json/test_images_triplets_dict.json"))
        for image_id in raw_gt_instance_dict:
            gt_dict[image_id]={}
            instance_dict=raw_gt_instance_dict[image_id]['instances']
            triplet_dict=raw_gt_triplet_dict[image_id]['triplets']
            gt_dict[image_id]['subject_class_ids']=[]
            gt_dict[image_id]['object_class_ids']=[]
            gt_dict[image_id]['subject_boxes']=[]
            gt_dict[image_id]['object_boxes']=[]
            gt_dict[image_id]['relation_ids']=[]
            for triplet_id in triplet_dict:
                gt_dict[image_id]['subject_class_ids'].append(instance_dict[str(triplet_dict[triplet_id]['subject_instance_id'])]['class_id'])
                gt_dict[image_id]['object_class_ids'].append(instance_dict[str(triplet_dict[triplet_id]['object_instance_id'])]['class_id'])
                gt_dict[image_id]['subject_boxes'].append(instance_dict[str(triplet_dict[triplet_id]['subject_instance_id'])]['box'])
                gt_dict[image_id]['object_boxes'].append(instance_dict[str(triplet_dict[triplet_id]['object_instance_id'])]['box'])
                gt_dict[image_id]['relation_ids'].append(triplet_dict[triplet_id]['relation_id'])
        json.dump(gt_dict,open("gt.json",'w'))
    else:
        gt_dict = json.load(open(args.gt_json))
    pred_dict = json.load(open(args.pred_json))
    num_img, metrics = compute_metrics(gt_dict, pred_dict, top_n, args.beta2, args.iou_thresh)

    print('-- Metrics on {} images --'.format(num_img))
    print('recall@{}: {:.2f}'.format(top_n, metrics['recall'] * 100))
    if top_n == 'α':
        print('n: {}, mAP@{}: {:.2f}'.format(metrics['n'], top_n, metrics['mAP'] * 100))
    else:
        print('precision@{}: {:.2f}'.format(top_n, metrics['precision'] * 100))
        print('Y@{}: {:.2f}'.format(top_n, metrics['Y'] * 100))
    # print('f-{}@{}: {:.2f}'.format(args.beta2, top_n, metrics['f'] * 100))