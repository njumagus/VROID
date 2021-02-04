# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import time
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.utils.torch_utils import SelfGCNLayer, OtherGCNLayer

RELATION_PAIR_HEAD_REGISTRY = Registry("RELATION_PAIR_HEAD")
RELATION_PAIR_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_pair_head(cfg):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.RELATION_HEADS.RELATION_PAIR_HEAD.NAME
    return RELATION_PAIR_HEAD_REGISTRY.get(name)(cfg)

@RELATION_PAIR_HEAD_REGISTRY.register()
class PairHead17(nn.Module): # 12 without gcn
    def __init__(self, cfg):
        super().__init__()
        self.device=cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})

        self.language_fc1 = nn.Linear(300, 512)
        self.language_ac1 = nn.ReLU()
        self.language_fc2 = nn.Linear(512, 512)
        self.language_ac2 = nn.ReLU()

        self.location_fc1 = nn.Linear(8, 64)
        self.location_ac1 = nn.ReLU()
        self.location_fc2 = nn.Linear(64, 512)
        self.location_ac2 = nn.ReLU()

        self.visual_fc1 = nn.Linear(1024 * 2, 1024)
        self.visual_ac1 = nn.ReLU()
        self.visual_fc2 = nn.Linear(1024, 512)
        self.visual_ac2 = nn.ReLU()

        self.pair_fc1 = nn.Linear(512 * 3, 512)
        self.pair_ac1 = nn.ReLU()

        self.pair_affectby_instance=OtherGCNLayer(512,256)

        self.pair_fc2 = nn.Linear(1024, 512)
        self.pair_ac2 = nn.ReLU()
        self.pair_fc3 = nn.Linear(512, 1)
        self.pair_ac3 = nn.Sigmoid()

    def forward(self, pair_features, pair_instances,pred_instances,relation_instance_features,training=True):
        losses = {}
        metrics={}
        pair_instance_nums=[len(pair_instance) for pair_instance in pair_instances]
        pair_locations_mix = torch.cat([pair_instance.pred_pair_locations for pair_instance in pair_instances])
        pair_features_mix = torch.cat(pair_features)
        subject_classes_mix = []
        object_classes_mix = []
        for pair_instance in pair_instances:
            subject_classes_mix.append(pair_instance.pred_pair_sub_classes-1)
            object_classes_mix.append(pair_instance.pred_pair_obj_classes-1)
        subject_classes_mix = torch.cat(subject_classes_mix).long()
        object_classes_mix = torch.cat(object_classes_mix).long()
        instance_embedding_mix = self.semantic_embed(subject_classes_mix)-self.semantic_embed(object_classes_mix)  # n,300

        language_feature = self.language_fc1(instance_embedding_mix)
        language_feature = self.language_ac1(language_feature)
        language_feature = self.language_fc2(language_feature)
        language_feature = self.language_ac2(language_feature)

        location_feature = self.location_fc1(pair_locations_mix)
        location_feature = self.location_ac1(location_feature)
        location_feature = self.location_fc2(location_feature)
        location_feature = self.location_ac2(location_feature)

        visual_feature = self.visual_fc1(pair_features_mix)
        visual_feature = self.visual_ac1(visual_feature)
        visual_feature = self.visual_fc2(visual_feature)
        visual_feature = self.visual_ac2(visual_feature)

        feature_mix = torch.cat([language_feature,location_feature, visual_feature], dim=1)
        pair_features_512 = self.pair_ac1(self.pair_fc1(feature_mix))

        # pair affect by instance
        pair_features_from_instance_512s = []
        for i in range(len(pair_instance_nums)):
            pred_pair_instance = pair_instances[i]
            pred_instance = pred_instances[i]
            pair_instance_attention = pred_pair_instance.pred_pair_instance_relate_matrix
            pair_features_from_instance_512 = self.pair_affectby_instance(relation_instance_features[i],
                                                                          pair_instance_attention)
            pair_features_from_instance_512s.append(pair_features_from_instance_512)
        pair_features_from_instance_512_mix = torch.cat(pair_features_from_instance_512s)
        update_pair_features_1024_mix = torch.cat([pair_features_512, pair_features_from_instance_512_mix], dim=1)

        pair_features_512 = self.pair_ac2(self.pair_fc2(update_pair_features_1024_mix))
        pair_interest_pred = self.pair_ac3(self.pair_fc3(pair_features_512)).squeeze(1)

        pair_interest_preds, pair_features_512s, losses, metrics = compute_pair_result(pair_interest_pred,
                                                                                       pair_features_512,
                                                                                       pair_instances, losses, metrics,
                                                                                       self.binary_focal_loss, training)
        return pair_interest_preds, pair_features_512s, losses, metrics

    def binary_focal_loss(self, pred, gt, pos_gamma=1.0,neg_gamma=2.0):
        # print("======================================")
        num_1=torch.sum(gt).item()*1.0
        num_0=gt.shape[0]-num_1
        alpha=0.5#1.0-num_1/gt.shape[0]
        # print(alpha)
        # print(pred)
        # print(gt)
        epsilon=1.e-5
        pred=pred.clamp(epsilon,1-epsilon)
        ce_1 = gt*(-torch.log(pred)) # gt=1
        ce_0 = (1-gt)*(-torch.log(1-pred)) # gt=0

        # ce=ce_1+ce_0
        # ce_avg=torch.mean(ce)
        # print(ce_0.shape)
        # print("ce_1")
        # print(ce_1)
        # print("ce_0")
        # print(ce_0)

        fl_1 = torch.pow(1-pred,pos_gamma)*ce_1
        # print("fl_1")
        # print(fl_1)
        # fl_1 = alpha*fl_1
        # print(fl_1)

        fl_0 = torch.pow(pred,neg_gamma)*ce_0
        # print("fl_0")
        # print(fl_0)
        # fl_0 = (1-alpha)*fl_0
        # print(fl_0)

        if num_1==0:
            fl_1_avg = torch.sum(fl_1)
        else:
            fl_1_avg = torch.sum(fl_1) / num_1
        if num_0==0:
            fl_0_avg = torch.sum(fl_0)
        else:
            fl_0_avg = torch.sum(fl_0)/ num_0
        # fl=fl_0+fl_1
        # fl_avg=torch.mean(fl)
        # print(fl_avg)
        # print(fl_1_avg)
        # print(fl_0_avg)
        # print("======================================")
        return fl_1_avg, fl_0_avg

@RELATION_PAIR_HEAD_REGISTRY.register()
class PairHead20(nn.Module): # 17 without semantic
    def __init__(self, cfg):
        super().__init__()
        self.device=cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        self.location_fc1 = nn.Linear(8, 64)
        self.location_ac1 = nn.ReLU()
        self.location_fc2 = nn.Linear(64, 512)
        self.location_ac2 = nn.ReLU()

        self.visual_fc1 = nn.Linear(1024 * 2, 1024)
        self.visual_ac1 = nn.ReLU()
        self.visual_fc2 = nn.Linear(1024, 512)
        self.visual_ac2 = nn.ReLU()

        self.pair_fc1 = nn.Linear(512 * 2, 512)
        self.pair_ac1 = nn.ReLU()

        self.pair_affectby_instance=OtherGCNLayer(512,256)

        self.pair_fc2 = nn.Linear(1024, 512)
        self.pair_ac2 = nn.ReLU()
        self.pair_fc3 = nn.Linear(512, 1)
        self.pair_ac3 = nn.Sigmoid()

    def forward(self, pair_features, pair_instances,pred_instances,relation_instance_features,training=True):
        losses = {}
        metrics={}
        pair_instance_nums=[len(pair_instance) for pair_instance in pair_instances]
        pair_locations_mix = torch.cat([pair_instance.pred_pair_locations for pair_instance in pair_instances])
        pair_features_mix = torch.cat(pair_features)
        subject_classes_mix = []
        object_classes_mix = []
        for pair_instance in pair_instances:
            subject_classes_mix.append(pair_instance.pred_pair_sub_classes-1)
            object_classes_mix.append(pair_instance.pred_pair_obj_classes-1)
        subject_classes_mix = torch.cat(subject_classes_mix).long()
        object_classes_mix = torch.cat(object_classes_mix).long()

        location_feature = self.location_fc1(pair_locations_mix)
        location_feature = self.location_ac1(location_feature)
        location_feature = self.location_fc2(location_feature)
        location_feature = self.location_ac2(location_feature)

        visual_feature = self.visual_fc1(pair_features_mix)
        visual_feature = self.visual_ac1(visual_feature)
        visual_feature = self.visual_fc2(visual_feature)
        visual_feature = self.visual_ac2(visual_feature)

        feature_mix = torch.cat([location_feature, visual_feature], dim=1)

        pair_features_512 = self.pair_fc1(feature_mix)
        pair_features_512_mix = self.pair_ac1(pair_features_512)

        # pair affect by instance
        pair_features_from_instance_512s = []
        for i in range(len(pair_instance_nums)):
            pred_pair_instance = pair_instances[i]
            pred_instance = pred_instances[i]
            pair_instance_attention = pred_pair_instance.pred_pair_instance_relate_matrix
            pair_features_from_instance_512 = self.pair_affectby_instance(relation_instance_features[i],
                                                                          pair_instance_attention)
            pair_features_from_instance_512s.append(pair_features_from_instance_512)
        pair_features_from_instance_512_mix = torch.cat(pair_features_from_instance_512s)
        update_pair_features_1024_mix = torch.cat([pair_features_512_mix, pair_features_from_instance_512_mix], dim=1)

        pair_features_512 = self.pair_ac2(self.pair_fc2(update_pair_features_1024_mix))
        pair_interest_pred = self.pair_ac3(self.pair_fc3(pair_features_512)).squeeze(1)

        pair_interest_preds, pair_features_512s, losses, metrics = compute_pair_result(pair_interest_pred,
                                                                                       pair_features_512,
                                                                                       pair_instances, losses, metrics,
                                                                                       self.binary_focal_loss, training)
        return pair_interest_preds, pair_features_512s, losses, metrics

    def binary_focal_loss(self, pred, gt, pos_gamma=1.0,neg_gamma=2.0):
        # print("======================================")
        num_1=torch.sum(gt).item()*1.0
        num_0=gt.shape[0]-num_1
        alpha=0.5#1.0-num_1/gt.shape[0]
        # print(alpha)
        # print(pred)
        # print(gt)
        epsilon=1.e-5
        pred=pred.clamp(epsilon,1-epsilon)
        ce_1 = gt*(-torch.log(pred)) # gt=1
        ce_0 = (1-gt)*(-torch.log(1-pred)) # gt=0

        # ce=ce_1+ce_0
        # ce_avg=torch.mean(ce)
        # print(ce_0.shape)
        # print("ce_1")
        # print(ce_1)
        # print("ce_0")
        # print(ce_0)

        fl_1 = torch.pow(1-pred,pos_gamma)*ce_1
        # print("fl_1")
        # print(fl_1)
        # fl_1 = alpha*fl_1
        # print(fl_1)

        fl_0 = torch.pow(pred,neg_gamma)*ce_0
        # print("fl_0")
        # print(fl_0)
        # fl_0 = (1-alpha)*fl_0
        # print(fl_0)

        if num_1==0:
            fl_1_avg = torch.sum(fl_1)
        else:
            fl_1_avg = torch.sum(fl_1) / num_1
        if num_0==0:
            fl_0_avg = torch.sum(fl_0)
        else:
            fl_0_avg = torch.sum(fl_0)/ num_0
        # fl=fl_0+fl_1
        # fl_avg=torch.mean(fl)
        # print(fl_avg)
        # print(fl_1_avg)
        # print(fl_0_avg)
        # print("======================================")
        return fl_1_avg, fl_0_avg

@RELATION_PAIR_HEAD_REGISTRY.register()
class PairHead21(nn.Module): # 17 without location
    def __init__(self, cfg):
        super().__init__()
        self.device=cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})

        self.language_fc1 = nn.Linear(300, 512)
        self.language_ac1 = nn.ReLU()
        self.language_fc2 = nn.Linear(512, 512)
        self.language_ac2 = nn.ReLU()

        self.visual_fc1 = nn.Linear(1024 * 2, 1024)
        self.visual_ac1 = nn.ReLU()
        self.visual_fc2 = nn.Linear(1024, 512)
        self.visual_ac2 = nn.ReLU()

        self.pair_fc1 = nn.Linear(512 * 2, 512)
        self.pair_ac1 = nn.ReLU()

        self.pair_affectby_instance=OtherGCNLayer(512,256)

        self.pair_fc2 = nn.Linear(1024, 512)
        self.pair_ac2 = nn.ReLU()
        self.pair_fc3 = nn.Linear(512, 1)
        self.pair_ac3 = nn.Sigmoid()

    def forward(self, pair_features, pair_instances,pred_instances,relation_instance_features,training=True):
        losses = {}
        metrics={}
        pair_instance_nums=[len(pair_instance) for pair_instance in pair_instances]
        pair_features_mix = torch.cat(pair_features)
        subject_classes_mix = []
        object_classes_mix = []
        for pair_instance in pair_instances:
            subject_classes_mix.append(pair_instance.pred_pair_sub_classes-1)
            object_classes_mix.append(pair_instance.pred_pair_obj_classes-1)
        subject_classes_mix = torch.cat(subject_classes_mix).long()
        object_classes_mix = torch.cat(object_classes_mix).long()
        instance_embedding_mix = self.semantic_embed(subject_classes_mix)-self.semantic_embed(object_classes_mix)  # n,300

        language_feature = self.language_fc1(instance_embedding_mix)
        language_feature = self.language_ac1(language_feature)
        language_feature = self.language_fc2(language_feature)
        language_feature = self.language_ac2(language_feature)

        visual_feature = self.visual_fc1(pair_features_mix)
        visual_feature = self.visual_ac1(visual_feature)
        visual_feature = self.visual_fc2(visual_feature)
        visual_feature = self.visual_ac2(visual_feature)

        feature_mix = torch.cat([language_feature, visual_feature], dim=1)

        pair_features_512 = self.pair_fc1(feature_mix)
        pair_features_512_mix = self.pair_ac1(pair_features_512)

        # pair affect by instance
        pair_features_from_instance_512s = []
        for i in range(len(pair_instance_nums)):
            pred_pair_instance = pair_instances[i]
            pred_instance = pred_instances[i]
            pair_instance_attention = pred_pair_instance.pred_pair_instance_relate_matrix
            pair_features_from_instance_512 = self.pair_affectby_instance(relation_instance_features[i],
                                                                          pair_instance_attention)
            pair_features_from_instance_512s.append(pair_features_from_instance_512)
        pair_features_from_instance_512_mix = torch.cat(pair_features_from_instance_512s)
        update_pair_features_1024_mix = torch.cat([pair_features_512_mix, pair_features_from_instance_512_mix], dim=1)

        pair_features_512 = self.pair_ac2(self.pair_fc2(update_pair_features_1024_mix))
        pair_interest_pred = self.pair_ac3(self.pair_fc3(pair_features_512)).squeeze(1)

        pair_interest_preds, pair_features_512s, losses, metrics = compute_pair_result(pair_interest_pred,
                                                                                       pair_features_512,
                                                                                       pair_instances, losses, metrics,
                                                                                       self.binary_focal_loss, training)
        return pair_interest_preds, pair_features_512s, losses, metrics

    def binary_focal_loss(self, pred, gt, pos_gamma=1.0,neg_gamma=2.0):
        # print("======================================")
        num_1=torch.sum(gt).item()*1.0
        num_0=gt.shape[0]-num_1
        alpha=0.5#1.0-num_1/gt.shape[0]
        # print(alpha)
        # print(pred)
        # print(gt)
        epsilon=1.e-5
        pred=pred.clamp(epsilon,1-epsilon)
        ce_1 = gt*(-torch.log(pred)) # gt=1
        ce_0 = (1-gt)*(-torch.log(1-pred)) # gt=0

        # ce=ce_1+ce_0
        # ce_avg=torch.mean(ce)
        # print(ce_0.shape)
        # print("ce_1")
        # print(ce_1)
        # print("ce_0")
        # print(ce_0)

        fl_1 = torch.pow(1-pred,pos_gamma)*ce_1
        # print("fl_1")
        # print(fl_1)
        # fl_1 = alpha*fl_1
        # print(fl_1)

        fl_0 = torch.pow(pred,neg_gamma)*ce_0
        # print("fl_0")
        # print(fl_0)
        # fl_0 = (1-alpha)*fl_0
        # print(fl_0)

        if num_1==0:
            fl_1_avg = torch.sum(fl_1)
        else:
            fl_1_avg = torch.sum(fl_1) / num_1
        if num_0==0:
            fl_0_avg = torch.sum(fl_0)
        else:
            fl_0_avg = torch.sum(fl_0)/ num_0
        # fl=fl_0+fl_1
        # fl_avg=torch.mean(fl)
        # print(fl_avg)
        # print(fl_1_avg)
        # print(fl_0_avg)
        # print("======================================")
        return fl_1_avg, fl_0_avg

@RELATION_PAIR_HEAD_REGISTRY.register()
class PairHead22(nn.Module): # 17 with bceloss
    def __init__(self, cfg):
        super().__init__()
        self.device=cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})

        self.language_fc1 = nn.Linear(300, 512)
        self.language_ac1 = nn.ReLU()
        self.language_fc2 = nn.Linear(512, 512)
        self.language_ac2 = nn.ReLU()

        self.location_fc1 = nn.Linear(8, 64)
        self.location_ac1 = nn.ReLU()
        self.location_fc2 = nn.Linear(64, 512)
        self.location_ac2 = nn.ReLU()

        self.visual_fc1 = nn.Linear(1024 * 2, 1024)
        self.visual_ac1 = nn.ReLU()
        self.visual_fc2 = nn.Linear(1024, 512)
        self.visual_ac2 = nn.ReLU()

        self.pair_fc1 = nn.Linear(512 * 3, 512)
        self.pair_ac1 = nn.ReLU()

        self.pair_affectby_instance=OtherGCNLayer(512,256)

        self.pair_fc2 = nn.Linear(1024, 512)
        self.pair_ac2 = nn.ReLU()
        self.pair_fc3 = nn.Linear(512, 1)
        self.pair_ac3 = nn.Sigmoid()

    def forward(self, pair_features, pair_instances,pred_instances,relation_instance_features,training=True):
        losses = {}
        metrics={}
        pair_instance_nums=[len(pair_instance) for pair_instance in pair_instances]
        pair_locations_mix = torch.cat([pair_instance.pred_pair_locations for pair_instance in pair_instances])
        pair_features_mix = torch.cat(pair_features)
        subject_classes_mix = []
        object_classes_mix = []
        for pair_instance in pair_instances:
            subject_classes_mix.append(pair_instance.pred_pair_sub_classes-1)
            object_classes_mix.append(pair_instance.pred_pair_obj_classes-1)
        subject_classes_mix = torch.cat(subject_classes_mix).long()
        object_classes_mix = torch.cat(object_classes_mix).long()
        instance_embedding_mix = self.semantic_embed(subject_classes_mix)-self.semantic_embed(object_classes_mix)  # n,300

        language_feature = self.language_fc1(instance_embedding_mix)
        language_feature = self.language_ac1(language_feature)
        language_feature = self.language_fc2(language_feature)
        language_feature = self.language_ac2(language_feature)

        location_feature = self.location_fc1(pair_locations_mix)
        location_feature = self.location_ac1(location_feature)
        location_feature = self.location_fc2(location_feature)
        location_feature = self.location_ac2(location_feature)

        visual_feature = self.visual_fc1(pair_features_mix)
        visual_feature = self.visual_ac1(visual_feature)
        visual_feature = self.visual_fc2(visual_feature)
        visual_feature = self.visual_ac2(visual_feature)

        feature_mix = torch.cat([language_feature,location_feature, visual_feature], dim=1)

        pair_features_512 = self.pair_fc1(feature_mix)
        pair_features_512_mix = self.pair_ac1(pair_features_512)

        # pair affect by instance
        pair_features_from_instance_512s = []
        for i in range(len(pair_instance_nums)):
            pred_pair_instance = pair_instances[i]
            pred_instance = pred_instances[i]
            pair_instance_attention = pred_pair_instance.pred_pair_instance_relate_matrix
            pair_features_from_instance_512 = self.pair_affectby_instance(relation_instance_features[i],
                                                                          pair_instance_attention)
            pair_features_from_instance_512s.append(pair_features_from_instance_512)
        pair_features_from_instance_512_mix = torch.cat(pair_features_from_instance_512s)
        update_pair_features_1024_mix = torch.cat([pair_features_512_mix, pair_features_from_instance_512_mix], dim=1)

        pair_features_512 = self.pair_ac2(self.pair_fc2(update_pair_features_1024_mix))
        pair_interest_pred = self.pair_ac3(self.pair_fc3(pair_features_512)).squeeze(1)

        pair_interest_preds, pair_features_512s, losses, metrics = compute_pair_result_bce(pair_interest_pred,
                                                                                       pair_features_512,
                                                                                       pair_instances, losses, metrics,
                                                                                       F.binary_cross_entropy, training)
        return pair_interest_preds, pair_features_512s, losses, metrics

    def binary_focal_loss(self, pred, gt, pos_gamma=1.0,neg_gamma=2.0):
        # print("======================================")
        num_1=torch.sum(gt).item()*1.0
        num_0=gt.shape[0]-num_1
        alpha=0.5#1.0-num_1/gt.shape[0]
        # print(alpha)
        # print(pred)
        # print(gt)
        epsilon=1.e-5
        pred=pred.clamp(epsilon,1-epsilon)
        ce_1 = gt*(-torch.log(pred)) # gt=1
        ce_0 = (1-gt)*(-torch.log(1-pred)) # gt=0

        # ce=ce_1+ce_0
        # ce_avg=torch.mean(ce)
        # print(ce_0.shape)
        # print("ce_1")
        # print(ce_1)
        # print("ce_0")
        # print(ce_0)

        fl_1 = torch.pow(1-pred,pos_gamma)*ce_1
        # print("fl_1")
        # print(fl_1)
        # fl_1 = alpha*fl_1
        # print(fl_1)

        fl_0 = torch.pow(pred,neg_gamma)*ce_0
        # print("fl_0")
        # print(fl_0)
        # fl_0 = (1-alpha)*fl_0
        # print(fl_0)

        if num_1==0:
            fl_1_avg = torch.sum(fl_1)
        else:
            fl_1_avg = torch.sum(fl_1) / num_1
        if num_0==0:
            fl_0_avg = torch.sum(fl_0)
        else:
            fl_0_avg = torch.sum(fl_0)/ num_0
        # fl=fl_0+fl_1
        # fl_avg=torch.mean(fl)
        # print(fl_avg)
        # print(fl_1_avg)
        # print(fl_0_avg)
        # print("======================================")
        return fl_1_avg, fl_0_avg

def compute_pair_result(pair_interest_pred_mix, pair_features_512_mix, pair_instances, losses, metrics, loss_func,training):
    pair_instance_nums=[len(pair_instance) for pair_instance in pair_instances]
    pair_interest_preds = pair_interest_pred_mix.split(pair_instance_nums)
    pair_features_512s = pair_features_512_mix.split(pair_instance_nums)

    print("pair")
    # print(pair_interest_preds)
    if training:
        pair_interest_pred_gts = [pair_instance.pred_pair_interest.float() for pair_instance in pair_instances]
        pair_interest_pred_gt_mix = torch.cat(pair_interest_pred_gts)
        losses["pair_pos_loss"], losses["pair_neg_loss"] = loss_func(pair_interest_pred_mix,pair_interest_pred_gt_mix)

        # print(pair_interest_pred_gts)
        tps = 0
        ps = 0
        gs = 0
        for i in range(len(pair_instances)):
            pair_interest_pred_gt = pair_interest_pred_gts[i]
            pair_interest_pred = pair_interest_preds[i]
            k = int(torch.sum(pair_interest_pred_gt).item())
            if k > pair_interest_pred.shape[0]:
                k = pair_interest_pred.shape[0]
                print(k)
            if k > 0:
                pair_interest_pred_score, pair_interest_pred_index = torch.topk(pair_interest_pred, k)
                print(pair_interest_pred_score)
                print(str(k) + " " + str(torch.sum(pair_interest_pred >= pair_interest_pred_score[-1].item()).item()))
                pair_interest_pred_pred = torch.where(pair_interest_pred >= pair_interest_pred_score[-1].item(),
                                                      torch.ones_like(pair_interest_pred),
                                                      torch.zeros_like(pair_interest_pred))
                tp = torch.sum(pair_interest_pred_pred * pair_interest_pred_gt)
                p = torch.sum(pair_interest_pred_pred)
            else:
                tp = 0
                p = 0
            g = torch.sum(pair_interest_pred_gt)
            tps += tp
            ps += p
            gs += g
        # print("pair-tp:"+str(tp)+" g:"+str(g))
        metrics['pair_tp'] = tps
        metrics['pair_p'] = ps
        metrics['pair_g'] = gs
    return pair_interest_preds,pair_features_512s,losses,metrics

def compute_pair_result_bce(pair_interest_pred_mix, pair_features_512_mix, pair_instances, losses, metrics, loss_func,training):
    pair_instance_nums=[len(pair_instance) for pair_instance in pair_instances]
    pair_interest_preds = pair_interest_pred_mix.split(pair_instance_nums)
    pair_features_512s = pair_features_512_mix.split(pair_instance_nums)

    print("pair")
    # print(pair_interest_preds)
    if training:
        pair_interest_pred_gts = [pair_instance.pred_pair_interest.float() for pair_instance in pair_instances]
        pair_interest_pred_gt_mix = torch.cat(pair_interest_pred_gts)
        losses["pair_pos_loss"] = loss_func(pair_interest_pred_mix,pair_interest_pred_gt_mix)

        # print(pair_interest_pred_gts)
        tps = 0
        ps = 0
        gs = 0
        for i in range(len(pair_instances)):
            pair_interest_pred_gt = pair_interest_pred_gts[i]
            pair_interest_pred = pair_interest_preds[i]
            k = int(torch.sum(pair_interest_pred_gt).item())
            if k > pair_interest_pred.shape[0]:
                k = pair_interest_pred.shape[0]
                print(k)
            if k > 0:
                pair_interest_pred_score, pair_interest_pred_index = torch.topk(pair_interest_pred, k)
                print(pair_interest_pred_score)
                print(str(k) + " " + str(torch.sum(pair_interest_pred >= pair_interest_pred_score[-1].item()).item()))
                pair_interest_pred_pred = torch.where(pair_interest_pred >= pair_interest_pred_score[-1].item(),
                                                      torch.ones_like(pair_interest_pred),
                                                      torch.zeros_like(pair_interest_pred))
                tp = torch.sum(pair_interest_pred_pred * pair_interest_pred_gt)
                p = torch.sum(pair_interest_pred_pred)
            else:
                tp = 0
                p = 0
            g = torch.sum(pair_interest_pred_gt)
            tps += tp
            ps += p
            gs += g
        # print("pair-tp:"+str(tp)+" g:"+str(g))
        metrics['pair_tp'] = tps
        metrics['pair_p'] = ps
        metrics['pair_g'] = gs
    return pair_interest_preds,pair_features_512s,losses,metrics