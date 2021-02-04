# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

RELATION_INSTANCE_ENCODER_REGISTRY = Registry("RELATION_INSTANCE_ENCODER")
RELATION_INSTANCE_ENCODER_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_instance_encoder(cfg):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.RELATION_HEADS.RELATION_INSTANCE_ENCODER.NAME
    return RELATION_INSTANCE_ENCODER_REGISTRY.get(name)(cfg)

@RELATION_INSTANCE_ENCODER_REGISTRY.register()
class InstanceEncoder1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mask_on = cfg.MODEL.RELATION_HEADS.MASK_ON
        self.instance_num=cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        self.cls_fc = nn.Linear(1024,1024)
        self.cls_score = nn.Linear(1024, self.instance_num)
        self.cls_ac = nn.Softmax(1)
        self.cls_loss_func = nn.CrossEntropyLoss()

        if self.mask_on:
            self.mask_fc1 = nn.Linear(128*128,64*64)
            self.mask_fc2 = nn.Linear(64 * 64, 32 * 32)
            self.feature_fc1 = nn.Linear(1024*2, 1024)

    def forward(self, image_features, pred_instances, pred_instance_box_features,training=True):
        losses={}
        metrics={}
        pred_instance_nums=[len(pred_instance) for pred_instance in pred_instances]

        pred_instance_box_features_mix=torch.cat(pred_instance_box_features)
        pred_instance_box_features_mix=F.relu(self.cls_fc(pred_instance_box_features_mix))
        pred_instance_probs_mix=self.cls_score(pred_instance_box_features_mix)
        pred_instance_logits_mix=self.cls_ac(pred_instance_probs_mix)
        if training:
            pred_classes_mix = torch.cat([pred_instance.pred_classes.long() for pred_instance in pred_instances])
            pred_gt_classes_mix = torch.cat([pred_instance.pred_gt_classes.long() for pred_instance in pred_instances])
            losses['pred_class_loss']=self.cls_loss_func(pred_instance_logits_mix,pred_gt_classes_mix)

            pred_instance_score_mix=torch.argmax(pred_instance_logits_mix,dim=1).long()
            pred_gt_tp=(pred_instance_score_mix==pred_gt_classes_mix)
            metrics['pred_class_tp']=torch.sum(pred_gt_tp).item()
            metrics['pred_class_p'] =pred_gt_classes_mix.shape[0]
            pred_tp = (pred_classes_mix == pred_gt_classes_mix)
            metrics['raw_pred_class_tp'] = torch.sum(pred_tp).item()
            metrics['raw_pred_class_p'] = pred_gt_classes_mix.shape[0]
        # if self.mask_on:
        #     pred_masks = pred_instances.pred_masks
        #     mask_result = F.relu(self.mask_fc1(pred_masks.view(pred_masks.shape[0],-1))) # n,128,128->n,128*128->n,64*64
        #     pred_instance_mask_features = F.relu(self.mask_fc2(mask_result)) # n,64*64->n,32*3
        #     pred_instance_features=torch.cat([pred_instance_box_features,pred_instance_mask_features],dim=1)
        #     pred_instance_features=self.feature_fc1(pred_instance_features)
        # else:
        #     pred_instance_features=pred_instance_box_features

        pred_instance_box_features=pred_instance_box_features_mix.split(pred_instance_nums)
        pred_instance_logits=pred_instance_logits_mix.split(pred_instance_nums)
        return pred_instance_box_features,pred_instance_logits,losses, metrics

@RELATION_INSTANCE_ENCODER_REGISTRY.register()
class InstanceEncoder2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mask_on = cfg.MODEL.RELATION_HEADS.MASK_ON
        self.instance_num=cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

        self.obj_ctx_rnn = nn.LSTM(1024, 512, num_layers=1, bidirectional=True)
        self.cls_fc = nn.Linear(1024 * 2, 1024)
        self.decoder_rnn = nn.LSTM(1024, self.instance_num, num_layers=1)

        self.cls_ac = nn.Softmax(1)
        self.cls_loss_func = nn.CrossEntropyLoss()

    def forward(self, image_features, pred_instances, pred_instance_box_features,training=True):
        losses={}
        metrics={}
        pred_instance_nums=[len(pred_instance) for pred_instance in pred_instances]

        instance_feature_1024s=[]
        pred_instance_logits=[]
        for i in range(len(pred_instances)):
            pred_instance=pred_instances[i]
            pred_instance_box_feature=pred_instance_box_features[i]
            instance_context, _ = self.obj_ctx_rnn(pred_instance_box_feature.unsqueeze(0)) # n,512
            instance_representation_with_context = torch.cat([pred_instance_box_feature,instance_context.squeeze(0)],dim=1) # n,1024+300+512
            instance_feature_1024 = self.cls_fc(instance_representation_with_context)
            instance_feature_1024s.append(instance_feature_1024)
            instance_dist,_ = self.decoder_rnn(instance_feature_1024.unsqueeze(0))
            instance_logit=self.cls_ac(instance_dist.squeeze(0))
            pred_instance_logits.append(instance_logit)
        pred_instance_logits_mix=torch.cat(pred_instance_logits)

        if training:
            pred_classes_mix = torch.cat([pred_instance.pred_classes.long() for pred_instance in pred_instances])
            pred_gt_classes_mix = torch.cat([pred_instance.pred_gt_classes.long() for pred_instance in pred_instances])
            losses['pred_class_loss']=self.cls_loss_func(pred_instance_logits_mix,pred_gt_classes_mix)

            pred_instance_score_mix=torch.argmax(pred_instance_logits_mix,dim=1).long()
            tp=(pred_instance_score_mix==pred_gt_classes_mix)
            metrics['pred_class_tp']=torch.sum(tp).item()
            metrics['pred_class_p'] =pred_gt_classes_mix.shape[0]
            pred_tp = (pred_classes_mix == pred_gt_classes_mix)
            metrics['raw_pred_class_tp'] = torch.sum(pred_tp).item()
            metrics['raw_pred_class_p'] = pred_gt_classes_mix.shape[0]
        return instance_feature_1024s,pred_instance_logits,losses, metrics

@RELATION_INSTANCE_ENCODER_REGISTRY.register()
class InstanceEncoder3(nn.Module): # no class prediction
    def __init__(self, cfg):
        super().__init__()
        self.mask_on = cfg.MODEL.RELATION_HEADS.MASK_ON
        self.instance_num=cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.cls_fc = nn.Linear(1024 , 1024)

    def forward(self, image_features, pred_instances, pred_instance_box_features, training=True):
        losses={}
        metrics={}

        update_pred_instance_box_features=[]
        for i in range(len(pred_instances)):
            pred_instance_box_feature=pred_instance_box_features[i]
            update_pred_instance_box_feature=self.cls_fc(pred_instance_box_feature)
            update_pred_instance_box_features.append(update_pred_instance_box_feature)
        return update_pred_instance_box_features,None,losses, metrics

@RELATION_INSTANCE_ENCODER_REGISTRY.register()
class InstanceEncoder4(nn.Module): # no instance encoder
    def __init__(self, cfg):
        super().__init__()
        self.mask_on = cfg.MODEL.RELATION_HEADS.MASK_ON
        self.instance_num=cfg.MODEL.RELATION_HEADS.INSTANCE_NUM

    def forward(self, image_features, pred_instances, pred_instance_box_features, training=True):
        losses={}
        metrics={}

        return pred_instance_box_features,None,losses, metrics