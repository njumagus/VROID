# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import time
import logging
import numpy as np
from matplotlib import pyplot as plt

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from fvcore.nn import smooth_l1_loss

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, Triplets
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.utils.torch_utils import box_iou

from .instance_encoder import build_instance_encoder
from .instance_head import build_instance_head
from .predicate_head import build_predicate_head
from .pair_head import build_pair_head
from .triplet_head import build_triplet_head

RELATION_HEADS_REGISTRY = Registry("RELATION_HEADS")
RELATION_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_relation_heads(cfg):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.RELATION_HEADS.NAME
    return RELATION_HEADS_REGISTRY.get(name)(cfg)

class RelationHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg):
        super(RelationHeads, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST

        if "instance" in self.relation_head_list:
            self._init_instance_head(cfg)
        if "pair" in self.relation_head_list:
            self._init_pair_head(cfg)
        if "predicate" in self.relation_head_list:
            self._init_predicate_head(cfg)
        if "triplet" in self.relation_head_list:
            self._init_triplet_head(cfg)

    def _init_instance_head(self, cfg):
        self.instance_head = build_instance_head(cfg)

    def _init_pair_head(self, cfg):
        self.pair_head = build_pair_head(cfg)

    def _init_predicate_head(self, cfg):
        self.predicate_head = build_predicate_head(cfg)

    def _init_triplet_head(self, cfg):
        self.triplet_head = build_triplet_head(cfg)

    def forward(self, panoptic_ins, triplets):
        raise NotImplementedError()


@RELATION_HEADS_REGISTRY.register()
class StandardRelationHeads(RelationHeads):

    def __init__(self, cfg):
        super(StandardRelationHeads, self).__init__(cfg)

    def forward(self, image_features, pred_instances, pred_pair_instances,
                      pred_instance_features, pred_pair_instance_features, pred_pair_predicate_features,
                      mannual_triplets,training=True,iteration=1):
        losses = {}
        metrics = {}
        results = {}

        instance_interest_features=None
        instance_interest_pred=None
        if "instance" in self.relation_head_list:
            instance_interest_pred, instance_interest_features, instance_loss, instance_metric = self.instance_head(pred_instance_features, pred_instances, pred_pair_instances,training)
            results['instance_interest_pred']=instance_interest_pred
            if training:
                losses.update(instance_loss)
                metrics.update(instance_metric)
            # print(instance_interest_pred.shape)
            # print(instance_interest_features[0].shape)

        pair_interest_features=None
        pair_interest_pred=None
        if "pair" in self.relation_head_list: # GCN - interest passing / iou effect
            pair_interest_pred, pair_interest_features, pair_loss, pair_metric = self.pair_head(pred_pair_predicate_features, pred_pair_instances, pred_instances, instance_interest_features,training)
            results['pair_interest_pred'] = pair_interest_pred
            if training:
                losses.update(pair_loss)
                metrics.update(pair_metric)
            # print(pair_interest_pred.shape)
            # print(pair_interest_features.shape)

        relation_predicate_features=None
        if "predicate" in self.relation_head_list:
            predicate_confidence, predicate_categories, pair_location,relation_predicate_features, predicate_loss, predicate_metric = self.predicate_head(
                pred_instances, pred_instance_features, instance_interest_pred,instance_interest_features,
                pred_pair_instances, pred_pair_predicate_features, pair_interest_pred,pair_interest_features,
                mannual_triplets, training, iteration)
            # print(predicate_confidence)
            # print(predicate_categories)
            # print(predicate_confidence.shape)
            # print(predicate_categories.shape)
            for i in range(len(pred_pair_instances)):
                pred_pair_instance=pred_pair_instances[i]
                predicate_category=predicate_categories[i]
                pred_pair_instance.pred_predicate_logits=predicate_category
                # pred_pair_instance.pred_predicate_classes=torch.argmax(predicate_category,dim=1)+1
            if predicate_confidence is not None:
                results['predicate_confidence'] = predicate_confidence
                # print(predicate_confidence.shape)
            if predicate_categories is not None:
                results['predicate_categories'] = predicate_categories
            # print(predicate_categories.shape)
            if pair_location is not None:
                results['pair_location']=pair_location
            if training:
                losses.update(predicate_loss)
                metrics.update(predicate_metric)

        if "triplet" in self.relation_head_list: # suppression
            triplet_interest_pred, triplet_predicates, triplet_loss, triplet_metric = self.triplet_head(pred_instances,pred_pair_instances,relation_predicate_features,
                                                                                    pair_interest_features,
                                                                                    training,iteration)
            if triplet_interest_pred is not None:
                results['triplet_interest_pred'] = triplet_interest_pred
            if triplet_predicates is not None:
                results['triplet_predicate'] = triplet_predicates
            if self.training:
                losses.update(triplet_loss)
                metrics.update(triplet_metric)

        return results, losses, metrics


# @RELATION_HEADS_REGISTRY.register()
# class TwoStepRelationHeads(RelationHeads):
#
#     def __init__(self, cfg):
#         super(TwoStepRelationHeads,self).__init__(cfg)
#
#     def forward(self,pred_instances, pred_pair_instances,
#                 pred_instance_features, pred_pair_instance_features, pred_pair_predicate_features,
#                 mannual_triplets, training=True,iteration=1):
#         losses = {}
#         metrics = {}
#         results = {}
#         instance_num = len(pred_instances)
#
#         if "instance" in self.relation_head_list:
#             instance_interest_pred, instance_interest_features, instance_loss, instance_metric = self.instance_head(
#                 pred_instance_features, pred_instances, pred_pair_instances,training)
#             results['instance_interest_pred'] = instance_interest_pred
#             if training:
#                 losses.update(instance_loss)
#                 metrics.update(instance_metric)
#
#         if "pair" in self.relation_head_list:  # GCN - interest passing / iou effect
#             pair_interest_pred, pair_interest_features, pair_loss, pair_metric = self.pair_head(
#                 pred_pair_predicate_features, pred_pair_instances, pred_instances, instance_interest_features,training)
#             results['pair_interest_pred'] = pair_interest_pred
#             if training:
#                 losses.update(pair_loss)
#                 metrics.update(pair_metric)
#
#         if "predicate" in self.relation_head_list:
#             predicate_confidence, predicate_categories, pair_location,relation_predicate_features, predicate_loss, predicate_metric = self.predicate_head(
#                 pred_instances, pred_instance_features, instance_interest_pred,
#                 pred_pair_instances, pred_pair_predicate_features, pair_interest_pred,
#                 mannual_triplets,training,iteration)
#             results['predicate_confidence'] = predicate_confidence
#             results['predicate_categories'] = predicate_categories
#             if pair_location is not None:
#                 results['pair_location'] = pair_location
#             if training:
#                 losses.update(predicate_loss)
#                 metrics.update(predicate_metric)
#
#
#         # if "triplet" in self.relation_head_list:  # suppression
#         #     if "instance" in self.relation_head_list:
#         #         # print("instance_interest_pred")
#         #         # print(instance_interest_pred)
#         #         triplet_interest_pred_of_instance = instance_interest_pred
#         #         # print(pair_interest_pred_from_instance.data.cpu().numpy())
#         #     else:
#         #         triplet_interest_pred_of_instance = None
#         #         instance_interest_features = None
#         #
#         #     if "pair" in self.relation_head_list:
#         #         eye = torch.eye(instance_interest_pred.shape[0]).flatten().to(self.device)
#         #         triplet_interest_pred_from_pair = pair_interest_pred * eye
#         #         # print(triplet_interest_pred_from_pair.squeeze(1).data.cpu().numpy())
#         #     else:
#         #         triplet_interest_pred_from_pair = None
#         #         pair_interest_features = None
#         #
#         #     if "predicate" in self.relation_head_list:
#         #         triplet_interest_pred_from_predicate_confidence = predicate_confidence
#         #         triplet_interest_pred_from_predicate = predicate_categories
#         #     else:
#         #         triplet_interest_pred_from_predicate_confidence = None
#         #         triplet_interest_pred_from_predicate = None
#         #         relation_predicate_features = None
#         #
#         #     triplet_interest_pred, triplet_loss, triplet_metric = self.triplet_head(pred_instances, pred_pair_instances,
#         #                                                                             pred_instance_features,
#         #                                                                             pred_pair_predicate_features,
#         #                                                                             # 1024, 2048
#         #                                                                             instance_interest_features,
#         #                                                                             pair_interest_features,
#         #                                                                             relation_predicate_features,
#         #                                                                             # 512,1024,1664
#         #                                                                             triplet_interest_pred_of_instance,
#         #                                                                             triplet_interest_pred_from_pair,
#         #                                                                             triplet_interest_pred_from_predicate_confidence,
#         #                                                                             triplet_interest_pred_from_predicate,
#         #                                                                             pred_gt_pair_predicate,
#         #                                                                             pred_gt_triplets_full, training)
#         #     results['triplet_interest_pred'] = triplet_interest_pred
#         #     if self.training:
#         #         losses.update(triplet_loss)
#         #     metrics.update(triplet_metric)
#
#         return results, losses, metrics

# @RELATION_HEADS_REGISTRY.register()
# class SemiRelationHeads(RelationHeads):
#
#     def __init__(self, cfg):
#         super(SemiRelationHeads, self).__init__(cfg)
#
#     def forward(self,
#                 pred_instances, pred_pair_instances,
#                 pred_instance_features, pred_pair_instance_features,pred_pair_predicate_features,
#                 mannual_triplets, training=True,iteration=1):
#         losses = {}
#         metrics = {}
#         results = {}
#         instance_num = len(pred_instances)
#
#         if "instance" in self.relation_head_list:
#             instance_interest_pred, instance_interest_features, instance_loss, instance_metric = self.instance_head(
#                 pred_instance_features, pred_instances, pred_gt_instances_full, pred_pair_instances,training)
#             results['instance_interest_pred'] = instance_interest_pred
#             if training:
#                 losses.update(instance_loss)
#             metrics.update(instance_metric)
#
#         if "pair" in self.relation_head_list:  # GCN - interest passing / iou effect
#             pair_interest_pred, pair_interest_features, pair_loss, pair_metric = self.pair_head(
#                 pred_pair_predicate_features, pred_pair_instances, pred_gt_pairs_full, pred_instances,
#                 instance_interest_features,training)
#             results['pair_interest_pred'] = pair_interest_pred
#             if training:
#                 losses.update(pair_loss)
#             metrics.update(pair_metric)
#
#         if "predicate" in self.relation_head_list:
#             if training:
#                 gt_predicate_confidence, gt_predicate_categories,gt_pair_location,gt_relation_predicate_features,gt_predicate_loss,gt_predicate_metric=self.predicate_head(
#                     gt_instances,gt_instance_features,gt_pair_instances,gt_pair_predicate_features,
#                     mannual_triplets,gt_pair_predicate,gt_triplets_full,training=True,iteration=iteration)
#                 results['gt_predicate_confidence'] = gt_predicate_confidence
#                 results['gt_predicate_categories'] = gt_predicate_categories
#                 losses.update(gt_predicate_loss)
#                 metrics.update(gt_predicate_metric)
#
#             else:
#                 predicate_confidence, predicate_categories, pair_location,relation_predicate_features, predicate_loss, predicate_metric = self.predicate_head(
#                     pred_instances, pred_instance_features, pred_pair_instances, pred_pair_predicate_features,
#                     mannual_triplets, pred_gt_pair_predicate, pred_gt_triplets_full, training=False)
#                 results['predicate_confidence'] = predicate_confidence
#                 results['predicate_categories'] = predicate_categories
#                 if pair_location is not None:
#                     results['pair_location'] = pair_location
#                 metrics.update(predicate_metric)
#
#         if "triplet" in self.relation_head_list:  # suppression
#             if "instance" in self.relation_head_list:
#                 # print("instance_interest_pred")
#                 # print(instance_interest_pred)
#                 triplet_interest_pred_of_instance = instance_interest_pred
#                 # print(pair_interest_pred_from_instance.data.cpu().numpy())
#             else:
#                 triplet_interest_pred_of_instance = None
#                 instance_interest_features = None
#
#             if "pair" in self.relation_head_list:
#                 eye = torch.eye(instance_interest_pred.shape[0]).flatten().to(self.device)
#                 triplet_interest_pred_from_pair = pair_interest_pred * eye
#                 # print(triplet_interest_pred_from_pair.squeeze(1).data.cpu().numpy())
#             else:
#                 triplet_interest_pred_from_pair = None
#                 pair_interest_features = None
#
#             if "predicate" in self.relation_head_list:
#                 triplet_interest_pred_from_predicate_confidence = predicate_confidence
#                 triplet_interest_pred_from_predicate = predicate_categories
#             else:
#                 triplet_interest_pred_from_predicate_confidence = None
#                 triplet_interest_pred_from_predicate = None
#                 relation_predicate_features = None
#
#             triplet_interest_pred, triplet_loss, triplet_metric = self.triplet_head(pred_instances, pred_pair_instances,
#                                                                                     pred_instance_features,
#                                                                                     pred_pair_predicate_features,
#                                                                                     # 1024, 2048
#                                                                                     instance_interest_features,
#                                                                                     pair_interest_features,
#                                                                                     relation_predicate_features,
#                                                                                     # 512,1024,1664
#                                                                                     triplet_interest_pred_of_instance,
#                                                                                     triplet_interest_pred_from_pair,
#                                                                                     triplet_interest_pred_from_predicate_confidence,
#                                                                                     triplet_interest_pred_from_predicate,
#                                                                                     pred_gt_pair_predicate,
#                                                                                     pred_gt_triplets_full, training)
#             results['triplet_interest_pred'] = triplet_interest_pred
#             if self.training:
#                 losses.update(triplet_loss)
#             metrics.update(triplet_metric)
#
#         return results, losses, metrics
