# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.utils.torch_utils import SelfGCNLayer, OtherGCNLayer
RELATION_TRIPLET_HEAD_REGISTRY = Registry("RELATION_TRIPLET_HEAD")
RELATION_TRIPLET_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_triplet_head(cfg):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.RELATION_HEADS.RELATION_TRIPLET_HEAD.NAME
    return RELATION_TRIPLET_HEAD_REGISTRY.get(name)(cfg)


def compute_triplet_result(pred_pair_instances,final_triplet_interest_pred_mix,loss_func,losses,metrics,training):
    pred_pair_instance_nums=[len(pred_pair_instance) for pred_pair_instance in pred_pair_instances]
    final_triplet_interest_preds=final_triplet_interest_pred_mix.split(pred_pair_instance_nums)

    print("triplet")
    # print(final_triplet_interest_preds)
    if training:
        final_triplet_interest_pred_gts = [pred_pair_instance.pred_pair_gt_predicate_full[:,1:] for pred_pair_instance in pred_pair_instances]
        # print(final_triplet_interest_pred_gts)
        final_triplet_interest_pred_gt_mix = torch.cat(final_triplet_interest_pred_gts)

        losses['triplet_interest_pos_loss'], losses['triplet_interest_neg_loss'] = loss_func(
            final_triplet_interest_pred_mix.flatten(), final_triplet_interest_pred_gt_mix.flatten())

        tps = 0
        tp20s = 0
        tp50s = 0
        tp100s = 0
        ps = 0
        p20s = 0
        p50s = 0
        p100s = 0
        gs = 0
        for i in range(len(pred_pair_instances)):
            predicate_pred = final_triplet_interest_preds[i].flatten()
            predicate_gt = final_triplet_interest_pred_gts[i].flatten()

            predicate_pred_score20, confidence_pred_index20 = torch.topk(predicate_pred, 20)
            predicate_pred_pred20 = torch.where(predicate_pred >= predicate_pred_score20[-1].item(),
                                                torch.ones_like(predicate_pred),
                                                torch.zeros_like(predicate_pred))
            tp20 = torch.sum(predicate_pred_pred20 * predicate_gt)
            p20 = torch.sum(predicate_pred_pred20)
            predicate_pred_score50, confidence_pred_index50 = torch.topk(predicate_pred, 50)
            predicate_pred_pred50 = torch.where(predicate_pred >= predicate_pred_score50[-1].item(),
                                                torch.ones_like(predicate_pred),
                                                torch.zeros_like(predicate_pred))
            tp50 = torch.sum(predicate_pred_pred50 * predicate_gt)
            p50 = torch.sum(predicate_pred_pred50)
            predicate_pred_score100, confidence_pred_index100 = torch.topk(predicate_pred, 100)
            predicate_pred_pred100 = torch.where(predicate_pred >= predicate_pred_score100[-1].item(),
                                                 torch.ones_like(predicate_pred),
                                                 torch.zeros_like(predicate_pred))
            tp100 = torch.sum(predicate_pred_pred100 * predicate_gt)
            p100 = torch.sum(predicate_pred_pred100)
            predicate_k = int(torch.sum(predicate_gt).item())
            if predicate_k > 0:
                predicate_pred_score, confidence_pred_index = torch.topk(predicate_pred, predicate_k)
                print(predicate_pred_score)
                print(str(predicate_k) + " " + str(
                    torch.sum(predicate_pred >= predicate_pred_score[-1].item()).item()))
                predicate_pred_pred = torch.where(predicate_pred >= predicate_pred_score[-1].item(),
                                                  torch.ones_like(predicate_pred),
                                                  torch.zeros_like(predicate_pred))
                tp = torch.sum(predicate_pred_pred * predicate_gt)
                p = torch.sum(predicate_pred_pred)
            else:
                tp = 0
                p = 0
            g = torch.sum(predicate_gt)
            tps += tp
            tp20s += tp20
            tp50s += tp50
            tp100s += tp100
            ps += p
            p20s += p20
            p50s += p50
            p100s += p100
            gs += g
        metrics['triplet_tp'] = tp
        metrics['triplet_tp20'] = tp20
        metrics['triplet_tp50'] = tp50
        metrics['triplet_tp100'] = tp100
        metrics['triplet_p'] = p
        metrics['triplet_p20'] = p20
        metrics['triplet_p50'] = p50
        metrics['triplet_p100'] = p100
        metrics['triplet_g'] = g

    return final_triplet_interest_preds,losses,metrics


@RELATION_TRIPLET_HEAD_REGISTRY.register()
class TripletHead4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.start_threshold = cfg.MODEL.RELATION_HEADS.RELATION_TRIPLET_HEAD.START_ITERATION_THRESHOLD

        # self.predicate_affectby_pair = OtherGCNLayer(512,512)
        # self.predicate_affectby_predicate = SelfGCNLayer(512, 512)

        self.interest_fc1 = nn.Linear(1024, 512)
        self.interest_ac1 = nn.ReLU()
        self.interest_fc2 = nn.Linear(512, self.relation_num-1)
        self.interest_ac2 = nn.Sigmoid()

    def forward(self, pred_instances, pred_pair_instances, relation_predicate_features,
                pair_interest_features,
                training=True, iteration=1):
        losses = {}
        metrics = {}

        if training and iteration < self.start_threshold:
            return None, None, losses, metrics

        pair_instance_nums = [len(pred_pair_instance) for pred_pair_instance in pred_pair_instances]
        pair_interest_feature_mix=torch.cat(pair_interest_features)
        relation_predicate_feature_mix=torch.cat(relation_predicate_features)
        triplet_feature_mix = torch.cat([pair_interest_feature_mix,relation_predicate_feature_mix],dim=1)

        triplet_feature_512_mix = self.interest_ac1(self.interest_fc1(triplet_feature_mix))
        triplet_interest_pred_mix = self.interest_ac2(self.interest_fc2(triplet_feature_512_mix))
        triplet_interest_preds = triplet_interest_pred_mix.split(pair_instance_nums)

        if training:
            triplet_interest_gts = [pred_pair_instance.pred_pair_gt_predicate_full[:,1:] for pred_pair_instance in pred_pair_instances]

        print("triplet")
        # print(triplet_interest_preds)
        if training:
            # print(triplet_interest_gts)
            # triplet_interest_pred_mix =
            triplet_interest_gt_mix = torch.cat(triplet_interest_gts)
            losses["triplet_pos_loss"], losses["triplet_neg_loss"] = self.binary_focal_loss(triplet_interest_pred_mix.flatten(),
                                                                                            triplet_interest_gt_mix.flatten())

            tps = 0
            ps = 0
            tp20s = 0
            p20s = 0
            tp50s = 0
            p50s = 0
            tp100s = 0
            p100s = 0
            gs = 0
            for i in range(len(pred_pair_instances)):
                predicate_gt = triplet_interest_gts[i].flatten()
                predicate_pred = triplet_interest_preds[i].flatten()

                predicate_k = int(torch.sum(predicate_gt).item())
                if predicate_k > 0:
                    if predicate_pred.shape[0] < 20:
                        k20 = predicate_pred.shape[0]
                    else:
                        k20 = 20
                    predicate_pred_score20, confidence_pred_index20 = torch.topk(predicate_pred, k20)
                    predicate_pred_pred20 = torch.where(predicate_pred >= predicate_pred_score20[-1].item(),
                                                        torch.ones_like(predicate_pred),
                                                        torch.zeros_like(predicate_pred))
                    tp20 = torch.sum(predicate_pred_pred20 * predicate_gt)
                    p20 = torch.sum(predicate_pred_pred20)

                    if predicate_pred.shape[0] < 50:
                        k50 = predicate_pred.shape[0]
                    else:
                        k50 = 50
                    predicate_pred_score50, confidence_pred_index50 = torch.topk(predicate_pred, k50)
                    predicate_pred_pred50 = torch.where(predicate_pred >= predicate_pred_score50[-1].item(),
                                                        torch.ones_like(predicate_pred),
                                                        torch.zeros_like(predicate_pred))
                    tp50 = torch.sum(predicate_pred_pred50 * predicate_gt)
                    p50 = torch.sum(predicate_pred_pred50)

                    if predicate_pred.shape[0] < 100:
                        k100 = predicate_pred.shape[0]
                    else:
                        k100 = 100
                    predicate_pred_score100, confidence_pred_index100 = torch.topk(predicate_pred, k100)
                    predicate_pred_pred100 = torch.where(predicate_pred >= predicate_pred_score100[-1].item(),
                                                         torch.ones_like(predicate_pred),
                                                         torch.zeros_like(predicate_pred))
                    tp100 = torch.sum(predicate_pred_pred100 * predicate_gt)
                    p100 = torch.sum(predicate_pred_pred100)

                    if predicate_pred.shape[0] < predicate_k:
                        predicate_k = predicate_pred.shape[0]
                    predicate_pred_score, confidence_pred_index = torch.topk(predicate_pred, predicate_k)
                    print(predicate_pred_score)
                    print(str(predicate_k) + " " + str(
                        torch.sum(predicate_pred >= predicate_pred_score[-1].item()).item()))
                    predicate_pred_pred = torch.where(predicate_pred >= predicate_pred_score[-1].item(),
                                                      torch.ones_like(predicate_pred), torch.zeros_like(predicate_pred))
                    tp = torch.sum(predicate_pred_pred * predicate_gt)
                    p = torch.sum(predicate_pred_pred)
                else:
                    tp = 0
                    p = 0
                    tp20 = 0
                    p20 = 0
                    tp50 = 0
                    p50 = 0
                    tp100 = 0
                    p100 = 0
                g = torch.sum(predicate_gt)
                tps += tp
                tp20s += tp20
                tp50s += tp50
                tp100s += tp100
                ps += p
                p20s += p20
                p50s += p50
                p100s += p100
                gs += g
                # print("instance-tp:" + str(tp) + " p:" + str(p) + " g:" + str(g))
            metrics['triplet_tp'] = tps
            metrics['triplet_tp20'] = tp20s
            metrics['triplet_tp50'] = tp50s
            metrics['triplet_tp100'] = tp100s
            metrics['triplet_p'] = ps
            metrics['triplet_p20'] = p20s
            metrics['triplet_p50'] = p50s
            metrics['triplet_p100'] = p100s
            metrics['triplet_g'] = gs
        return triplet_interest_preds, None, losses, metrics

    def binary_focal_loss(self, pred, gt, pos_gamma=1.0, neg_gamma=2.0):
        # print("======================================")
        num_1 = torch.sum(gt).item() * 1.0
        num_0 = gt.shape[0] - num_1
        alpha = 0.5  # 1.0-num_1/gt.shape[0]
        # print(alpha)
        # print(pred)
        # print(gt)
        epsilon = 1.e-5
        pred = pred.clamp(epsilon, 1 - epsilon)
        ce_1 = gt * (-torch.log(pred))  # gt=1
        ce_0 = (1 - gt) * (-torch.log(1 - pred))  # gt=0

        # ce=ce_1+ce_0
        # ce_avg=torch.mean(ce)
        # print(ce_0.shape)
        # print("ce_1")
        # print(ce_1)
        # print("ce_0")
        # print(ce_0)

        fl_1 = torch.pow(1 - pred, pos_gamma) * ce_1
        # print("fl_1")
        # print(fl_1)
        # fl_1 = alpha*fl_1
        # print(fl_1)

        fl_0 = torch.pow(pred, neg_gamma) * ce_0
        # print("fl_0")
        # print(fl_0)
        # fl_0 = (1-alpha)*fl_0
        # print(fl_0)

        if num_1 == 0:
            fl_1_avg = torch.sum(fl_1)
        else:
            fl_1_avg = torch.sum(fl_1) / num_1
        if num_0 == 0:
            fl_0_avg = torch.sum(fl_0)
        else:
            fl_0_avg = torch.sum(fl_0) / num_0
        # fl=fl_0+fl_1
        # fl_avg=torch.mean(fl)
        # print(fl_avg)
        # print(fl_1_avg)
        # print(fl_0_avg)
        # print("======================================")
        return fl_1_avg, fl_0_avg