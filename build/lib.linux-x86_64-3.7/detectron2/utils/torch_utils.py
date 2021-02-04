import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy.misc
import numpy as np
from detectron2.structures.triplets import Triplets

def extract_bbox(mask):
    horizontal_indicies = torch.where(torch.any(mask, dim=0))[0]
    vertical_indicies = torch.where(torch.any(mask, dim=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    box = torch.IntTensor([x1, y1, x2, y2])
    return box

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (x1, y1, x2, y2)].
    """
    boxes = torch.zeros([mask.shape[0], 4])
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        # Bounding box.
        horizontal_indicies = torch.where(torch.any(m, axis=0))[0]
        vertical_indicies = torch.where(torch.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i][0]=x1
        boxes[i][1]=y1
        boxes[i][2]=x2
        boxes[i][3]=y2
    return boxes.int()

def mask_iou(bool_mask_pred,bool_mask_gt):
    intersection = bool_mask_pred * bool_mask_gt
    union = bool_mask_pred + bool_mask_gt
    return np.count_nonzero(intersection) / np.count_nonzero(union)

def box_iou(box_pred, box_gt):
    area_pred=(box_pred[2]-box_pred[0])*(box_pred[3]-box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    area_sum = area_pred+area_gt

    left_line=max(box_pred[0],box_gt[0]) # x1
    top_line=max(box_pred[1],box_gt[1]) # y1
    right_line = min(box_pred[2], box_gt[2]) # x2
    bottom_line = min(box_pred[3], box_gt[3]) # y2

    if left_line>=right_line or top_line>=bottom_line:
        return 0
    else:
        intersect=(right_line-left_line)*(bottom_line-top_line)
        # print(str(left_line)+" "+str(right_line)+" "+str(top_line)+" "+str(bottom_line))
        # print(str(intersect)+" "+str(area_sum-intersect))
        # print()
        return ((intersect / (area_sum-intersect)) * 1.0).item()

def boxes_iou(box_preds, box_gts):
    area_preds=(box_preds[:,2]-box_preds[:,0])*(box_preds[:,3]-box_preds[:,1])
    area_gts = (box_gts[:,2] - box_gts[:,0]) * (box_gts[:,3] - box_gts[:,1])
    area_sums = area_preds+area_gts

    left_lines = torch.max(box_preds[:,0],box_gts[:,0]) # x1
    top_lines = torch.max(box_preds[:,1],box_gts[:,1]) # y1
    right_lines = torch.min(box_preds[:,2], box_gts[:,2]) # x2
    bottom_lines = torch.min(box_preds[:,3], box_gts[:,3]) # y2

    intersect_hs = right_lines - left_lines
    intersect_hs = torch.where(intersect_hs < 0, torch.zeros_like(intersect_hs), intersect_hs)
    intersect_ws = bottom_lines - top_lines
    intersect_ws = torch.where(intersect_ws < 0, torch.zeros_like(intersect_ws), intersect_ws)
    intersect = intersect_hs * intersect_ws
    return (intersect / (area_sums-intersect)) * 1.0

class SelfGCNLayer(nn.Module): # self
    def __init__(self,source_channel,impact_channel):
        super(SelfGCNLayer, self).__init__()
        # self.source_fc=nn.Linear(source_channel,source_channel)
        self.impact_fc=nn.Linear(impact_channel,source_channel)

    def forward(self,source,impact,attention): # [n1,x1] [n2,x2] [n1,n2]
        result = F.relu(self.impact_fc(impact)) # [n2,x2]->[n2,x3]
        collect = attention @ result # [n1,n2]@[n2,x3]=[n1,x3]
        collect_avg = collect / (attention.sum(1).view(collect.shape[0], 1) + 1e-7)
        update=(source+collect_avg)/2
        return update

class OtherGCNLayer(nn.Module): # other
    def __init__(self,output_channel,input_channel):
        super(OtherGCNLayer, self).__init__()
        # self.source_fc=nn.Linear(source_channel,source_channel)
        self.impact_fc=nn.Linear(input_channel,output_channel)

    def forward(self,impact,attention): # [n2,x2] [n1,n2]
        result = F.relu(self.impact_fc(impact)) # [n2,x2]->[n2,x3]
        collect = attention @ result # [n1,n2]@[n2,x3]=[n1,x3]
        collect_avg = collect / (attention.sum(1).view(collect.shape[0], 1) + 1e-7)
        return collect_avg