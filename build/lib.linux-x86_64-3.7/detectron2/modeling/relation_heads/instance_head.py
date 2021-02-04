# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from fvcore.nn import smooth_l1_loss

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.utils.torch_utils import SelfGCNLayer, OtherGCNLayer

RELATION_INSTANCE_HEAD_REGISTRY = Registry("RELATION_INSTANCE_HEAD")
RELATION_INSTANCE_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_instance_head(cfg):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.RELATION_HEADS.RELATION_INSTANCE_HEAD.NAME
    return RELATION_INSTANCE_HEAD_REGISTRY.get(name)(cfg)

@RELATION_INSTANCE_HEAD_REGISTRY.register()
class InstanceHead14(nn.Module): # 10 without gcn
    def __init__(self, cfg):
        super().__init__()
        self.device=cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})

        self.language_fc1 = nn.Linear(300,512)
        self.language_ac1 = nn.ReLU()
        self.language_fc2 = nn.Linear(512, 256)
        self.language_ac2 = nn.ReLU()

        self.location_fc1 = nn.Linear(4,32)
        self.location_ac1 = nn.ReLU()
        self.location_fc2 = nn.Linear(32, 256)
        self.location_ac2 = nn.ReLU()

        self.visual_fc1 = nn.Linear(1024,512)
        self.visual_ac1 = nn.ReLU()
        self.visual_fc2 = nn.Linear(512, 256)
        self.visual_ac2 = nn.ReLU()

        self.instance_fc1 = nn.Linear(256*3, 256)
        self.instance_ac1 = nn.Sigmoid()
        self.instance_fc2 = nn.Linear(256, 1)
        self.instance_ac2 = nn.Sigmoid()

    def forward(self, instance_features, pred_instances,pred_pair_instances,training=True):
        losses={}
        metrics={}
        pred_instance_nums=[len(pred_instance) for pred_instance in pred_instances]
        instance_locations_mix=torch.cat([pred_instance.pred_locations for pred_instance in pred_instances])
        instance_features_mix=torch.cat(instance_features)
        instance_class_mix=torch.cat([pred_instance.pred_classes-1 for pred_instance in pred_instances]).long()
        instance_embedding_mix = self.semantic_embed(instance_class_mix)  # n,300

        language_feature = self.language_fc1(instance_embedding_mix)
        language_feature = self.language_ac1(language_feature)
        language_feature = self.language_fc2(language_feature)
        language_feature = self.language_ac2(language_feature)

        location_feature = self.location_fc1(instance_locations_mix)
        location_feature = self.location_ac1(location_feature)
        location_feature = self.location_fc2(location_feature)
        location_feature = self.location_ac2(location_feature)

        visual_feature = self.visual_fc1(instance_features_mix)
        visual_feature = self.visual_ac1(visual_feature)
        visual_feature = self.visual_fc2(visual_feature)
        visual_feature = self.visual_ac2(visual_feature)

        feature_mix = torch.cat([language_feature,location_feature,visual_feature],dim=1)

        instance_features_256 = self.instance_ac1(self.instance_fc1(feature_mix))
        instance_interest_pred = self.instance_ac2(self.instance_fc2(instance_features_256)).squeeze(1)

        instance_interest_preds, instance_features_256s, losses, metrics=compute_instance_result(pred_instances,instance_interest_pred,instance_features_256,losses,metrics,self.binary_focal_loss,training)
        return instance_interest_preds, instance_features_256s, losses, metrics

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
        return fl_1_avg, fl_0_avg # location+focal loss

@RELATION_INSTANCE_HEAD_REGISTRY.register()
class InstanceHead15(nn.Module): # 14 without semantic
    def __init__(self, cfg):
        super().__init__()
        self.device=cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        self.location_fc1 = nn.Linear(4,32)
        self.location_ac1 = nn.ReLU()
        self.location_fc2 = nn.Linear(32, 256)
        self.location_ac2 = nn.ReLU()

        self.visual_fc1 = nn.Linear(1024,512)
        self.visual_ac1 = nn.ReLU()
        self.visual_fc2 = nn.Linear(512, 256)
        self.visual_ac2 = nn.ReLU()

        self.instance_fc1 = nn.Linear(256*2, 256)
        self.instance_ac1 = nn.ReLU()
        self.instance_fc2 = nn.Linear(256, 1)
        self.instance_ac2 = nn.Sigmoid()

    def forward(self, instance_features, pred_instances,pred_pair_instances,training=True):
        losses={}
        metrics={}
        pred_instance_nums=[len(pred_instance) for pred_instance in pred_instances]
        instance_locations_mix=torch.cat([pred_instance.pred_locations for pred_instance in pred_instances])
        instance_features_mix=torch.cat(instance_features)

        location_feature = self.location_fc1(instance_locations_mix)
        location_feature = self.location_ac1(location_feature)
        location_feature = self.location_fc2(location_feature)
        location_feature = self.location_ac2(location_feature)

        visual_feature = self.visual_fc1(instance_features_mix)
        visual_feature = self.visual_ac1(visual_feature)
        visual_feature = self.visual_fc2(visual_feature)
        visual_feature = self.visual_ac2(visual_feature)

        feature_mix = torch.cat([location_feature,visual_feature],dim=1)

        instance_features_256 = self.instance_fc1(feature_mix)
        instance_features_256 = self.instance_ac1(instance_features_256)
        instance_result = self.instance_fc2(instance_features_256)
        instance_interest_pred = self.instance_ac2(instance_result).squeeze(1)

        instance_interest_preds, instance_features_256s, losses, metrics=compute_instance_result(pred_instances,instance_interest_pred,instance_features_256,losses,metrics,self.binary_focal_loss,training)
        return instance_interest_preds, instance_features_256s, losses, metrics

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
        return fl_1_avg, fl_0_avg # location+focal loss

@RELATION_INSTANCE_HEAD_REGISTRY.register()
class InstanceHead16(nn.Module): # 14 without location
    def __init__(self, cfg):
        super().__init__()
        self.device=cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})

        self.language_fc1 = nn.Linear(300,512)
        self.language_ac1 = nn.ReLU()
        self.language_fc2 = nn.Linear(512, 256)
        self.language_ac2 = nn.ReLU()

        self.visual_fc1 = nn.Linear(1024,512)
        self.visual_ac1 = nn.ReLU()
        self.visual_fc2 = nn.Linear(512, 256)
        self.visual_ac2 = nn.ReLU()

        self.instance_fc1 = nn.Linear(256*2, 256)
        self.instance_ac1 = nn.ReLU()
        self.instance_fc2 = nn.Linear(256, 1)
        self.instance_ac2 = nn.Sigmoid()

    def forward(self, instance_features, pred_instances,pred_pair_instances,training=True):
        losses={}
        metrics={}
        pred_instance_nums=[len(pred_instance) for pred_instance in pred_instances]
        instance_features_mix=torch.cat(instance_features)
        instance_class_mix=torch.cat([pred_instance.pred_classes-1 for pred_instance in pred_instances]).long()
        instance_embedding_mix = self.semantic_embed(instance_class_mix)  # n,300

        language_feature = self.language_fc1(instance_embedding_mix)
        language_feature = self.language_ac1(language_feature)
        language_feature = self.language_fc2(language_feature)
        language_feature = self.language_ac2(language_feature)

        visual_feature = self.visual_fc1(instance_features_mix)
        visual_feature = self.visual_ac1(visual_feature)
        visual_feature = self.visual_fc2(visual_feature)
        visual_feature = self.visual_ac2(visual_feature)

        feature_mix = torch.cat([language_feature,visual_feature],dim=1)

        instance_features_256 = self.instance_fc1(feature_mix)
        instance_features_256 = self.instance_ac1(instance_features_256)
        instance_result = self.instance_fc2(instance_features_256)
        instance_interest_pred = self.instance_ac2(instance_result).squeeze(1)

        instance_interest_preds, instance_features_256s, losses, metrics=compute_instance_result(pred_instances,instance_interest_pred,instance_features_256,losses,metrics,self.binary_focal_loss,training)
        return instance_interest_preds, instance_features_256s, losses, metrics

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
        return fl_1_avg, fl_0_avg # location+focal loss

@RELATION_INSTANCE_HEAD_REGISTRY.register()
class InstanceHead17(nn.Module): # 14 with bceloss
    def __init__(self, cfg):
        super().__init__()
        self.device=cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})

        self.language_fc1 = nn.Linear(300,512)
        self.language_ac1 = nn.ReLU()
        self.language_fc2 = nn.Linear(512, 256)
        self.language_ac2 = nn.ReLU()

        self.location_fc1 = nn.Linear(4,32)
        self.location_ac1 = nn.ReLU()
        self.location_fc2 = nn.Linear(32, 256)
        self.location_ac2 = nn.ReLU()

        self.visual_fc1 = nn.Linear(1024,512)
        self.visual_ac1 = nn.ReLU()
        self.visual_fc2 = nn.Linear(512, 256)
        self.visual_ac2 = nn.ReLU()

        self.instance_fc1 = nn.Linear(256*3, 256)
        self.instance_ac1 = nn.ReLU()
        self.instance_fc2 = nn.Linear(256, 1)
        self.instance_ac2 = nn.Sigmoid()

    def forward(self, instance_features, pred_instances,pred_pair_instances,training=True):
        losses={}
        metrics={}
        pred_instance_nums=[len(pred_instance) for pred_instance in pred_instances]
        instance_locations_mix=torch.cat([pred_instance.pred_locations for pred_instance in pred_instances])
        instance_features_mix=torch.cat(instance_features)
        instance_class_mix=torch.cat([pred_instance.pred_classes-1 for pred_instance in pred_instances]).long()
        instance_embedding_mix = self.semantic_embed(instance_class_mix)  # n,300

        language_feature = self.language_fc1(instance_embedding_mix)
        language_feature = self.language_ac1(language_feature)
        language_feature = self.language_fc2(language_feature)
        language_feature = self.language_ac2(language_feature)

        location_feature = self.location_fc1(instance_locations_mix)
        location_feature = self.location_ac1(location_feature)
        location_feature = self.location_fc2(location_feature)
        location_feature = self.location_ac2(location_feature)

        visual_feature = self.visual_fc1(instance_features_mix)
        visual_feature = self.visual_ac1(visual_feature)
        visual_feature = self.visual_fc2(visual_feature)
        visual_feature = self.visual_ac2(visual_feature)

        feature_mix = torch.cat([language_feature,location_feature,visual_feature],dim=1)

        instance_features_256 = self.instance_fc1(feature_mix)
        instance_features_256 = self.instance_ac1(instance_features_256)
        instance_result = self.instance_fc2(instance_features_256)
        instance_interest_pred = self.instance_ac2(instance_result).squeeze(1)

        instance_interest_preds, instance_features_256s, losses, metrics=compute_instance_result_bce(pred_instances,instance_interest_pred,instance_features_256,losses,metrics,F.binary_cross_entropy,training)
        return instance_interest_preds, instance_features_256s, losses, metrics

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
        return fl_1_avg, fl_0_avg # location+focal loss

def compute_instance_result(pred_instances,instance_interest_pred_mix,instance_features_256_mix,losses,metrics,loss_func,training):
    pred_instance_nums = [len(pred_instance) for pred_instance in pred_instances]
    instance_interest_preds = instance_interest_pred_mix.split(pred_instance_nums)
    instance_features_256s = instance_features_256_mix.split(pred_instance_nums)

    print("instance")
    # print(instance_interest_preds)
    if training:
        instance_interest_pred_gts = [pred_instance.pred_interest.float() for pred_instance in pred_instances]
        instance_interest_pred_gt_mix = torch.cat(instance_interest_pred_gts)
        losses["instance_pos_loss"], losses["instance_neg_loss"] = loss_func(instance_interest_pred_mix, instance_interest_pred_gt_mix)

        # print(instance_interest_pred_gts)
        tps = 0
        ps = 0
        gs = 0
        for i in range(len(pred_instances)):
            instance_interest_pred_gt = instance_interest_pred_gts[i]
            instance_interest_pred = instance_interest_preds[i]
            k = int(torch.sum(instance_interest_pred_gt).item())
            if k > 0:
                instance_interest_pred_score, instance_interest_pred_index = torch.topk(instance_interest_pred, k)
                print(instance_interest_pred_score)
                print(str(k) + " " + str(torch.sum(instance_interest_pred >= instance_interest_pred_score[-1].item()).item()))
                instance_interest_pred_pred = torch.where(
                    instance_interest_pred >= instance_interest_pred_score[-1].item(),
                    torch.ones_like(instance_interest_pred), torch.zeros_like(instance_interest_pred))
                tp = torch.sum(instance_interest_pred_pred * instance_interest_pred_gt)
                p = torch.sum(instance_interest_pred_pred)
            else:
                tp = 0
                p = 0
            g = torch.sum(instance_interest_pred_gt)
            tps += tp
            ps += p
            gs += g
        # print("instance-tp:" + str(tp) + " p:" + str(p) + " g:" + str(g))
        metrics['instance_tp'] = tps
        metrics['instance_p'] = ps
        metrics['instance_g'] = gs
    return instance_interest_preds, instance_features_256s, losses, metrics

def compute_instance_result_bce(pred_instances,instance_interest_pred_mix,instance_features_256_mix,losses,metrics,loss_func,training):
    pred_instance_nums = [len(pred_instance) for pred_instance in pred_instances]
    instance_interest_preds = instance_interest_pred_mix.split(pred_instance_nums)
    instance_features_256s = instance_features_256_mix.split(pred_instance_nums)

    print("instance")
    # print(instance_interest_preds)
    if training:
        instance_interest_pred_gts = [pred_instance.pred_interest.float() for pred_instance in pred_instances]
        instance_interest_pred_gt_mix = torch.cat(instance_interest_pred_gts)
        losses["instance_pos_loss"] = loss_func(instance_interest_pred_mix, instance_interest_pred_gt_mix)

        # print(instance_interest_pred_gts)
        tps = 0
        ps = 0
        gs = 0
        for i in range(len(pred_instances)):
            instance_interest_pred_gt = instance_interest_pred_gts[i]
            instance_interest_pred = instance_interest_preds[i]
            k = int(torch.sum(instance_interest_pred_gt).item())
            if k > 0:
                instance_interest_pred_score, instance_interest_pred_index = torch.topk(instance_interest_pred, k)
                print(instance_interest_pred_score)
                print(str(k) + " " + str(torch.sum(instance_interest_pred >= instance_interest_pred_score[-1].item()).item()))
                instance_interest_pred_pred = torch.where(
                    instance_interest_pred >= instance_interest_pred_score[-1].item(),
                    torch.ones_like(instance_interest_pred), torch.zeros_like(instance_interest_pred))
                tp = torch.sum(instance_interest_pred_pred * instance_interest_pred_gt)
                p = torch.sum(instance_interest_pred_pred)
            else:
                tp = 0
                p = 0
            g = torch.sum(instance_interest_pred_gt)
            tps += tp
            ps += p
            gs += g
        # print("instance-tp:" + str(tp) + " p:" + str(p) + " g:" + str(g))
        metrics['instance_tp'] = tps
        metrics['instance_p'] = ps
        metrics['instance_g'] = gs
    return instance_interest_preds, instance_features_256s, losses, metrics