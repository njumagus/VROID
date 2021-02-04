# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from fvcore.nn import smooth_l1_loss

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.utils.torch_utils import SelfGCNLayer, OtherGCNLayer

RELATION_PREDICATE_HEAD_REGISTRY = Registry("RELATION_PREDICATE_HEAD")
RELATION_PREDICATE_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_predicate_head(cfg):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.NAME
    return RELATION_PREDICATE_HEAD_REGISTRY.get(name)(cfg)


@RELATION_PREDICATE_HEAD_REGISTRY.register()
class PredicateHeadsMFULN45(nn.Module):  # 4 with semi2
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.use_bias = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.USE_BIAS

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        # self.semantic_embed.weight.requires_grad=False

        self.lstm = nn.LSTM(300, 300, num_layers=1, batch_first=True)
        self.language_lstm_fc = nn.Linear(300, 512)

        self.language_fc = nn.Linear(512, 512)
        self.visual_fc = nn.Linear(2048, 512)
        self.union_location_fc = nn.Linear(8, 512)

        self.predicate_fc1 = nn.Linear(512 * 3, 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num - 1)
        self.predicate_ac2 = nn.Sigmoid()

        if self.use_bias:
            predicate_matrix = np.load("predicate_matrix.npy")
            predicate_matrix /= np.sum(predicate_matrix, 2)[:, :, None]
            self.freq_bias = FrequencyBias(cfg, predicate_matrix)

        self.unlabel_iteration_threshold1 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD1
        self.unlabel_iteration_threshold2 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD2
        self.af = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.AF

        self.predicate_loss_func = nn.BCELoss()

    def forward(self,
                pred_instances, pred_features, instance_interest_preds, relation_instance_features,
                pred_pair_instances, pred_pair_predicate_features, pair_interest_preds, relation_pair_features,
                mannual_triplets, training=True, iteration=1):
        losses = {}
        metrics = {}

        subobj_categories_mix = []
        pred_pair_predicate_features_mix = []
        pred_pair_union_locations_mix = []
        predicate_gts = []
        labeled_pair_locations = []
        unlabeled_pair_locations = []
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            sub_classes = pred_pair_instance.pred_pair_sub_classes - 1
            obj_classes = pred_pair_instance.pred_pair_obj_classes - 1
            pred_pair_predicate_features_this = pred_pair_predicate_features[i]
            pred_pair_union_locations_this = pred_pair_instance.pred_pair_union_locations
            subobj_categories = torch.cat([sub_classes.unsqueeze(1), obj_classes.unsqueeze(1)], dim=1).long()
            subobj_categories_mix.append(subobj_categories)
            pred_pair_predicate_features_mix.append(pred_pair_predicate_features_this)
            pred_pair_union_locations_mix.append(pred_pair_union_locations_this)
            if training:
                labeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest > 0))
                unlabeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest <= 0))
                predicate_gts.append(pred_pair_instance.pred_pair_gt_predicate_full[:, 1:])

        subobj_categories_mix = torch.cat(subobj_categories_mix)
        pred_pair_predicate_features_mix = torch.cat(pred_pair_predicate_features_mix)
        pred_pair_union_locations_mix = torch.cat(pred_pair_union_locations_mix)

        subobj_embeddings = self.semantic_embed(subobj_categories_mix)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(subobj_embeddings)
        language_lstm_out = self.language_lstm_fc(lstm_out[:, -1, :])
        language_out = F.relu(self.language_fc(language_lstm_out))

        visual_out = F.relu(self.visual_fc(pred_pair_predicate_features_mix))
        union_location_out = F.relu(self.union_location_fc(pred_pair_union_locations_mix))

        features = torch.cat([language_out, visual_out, union_location_out], dim=1)

        predicate_feature = self.predicate_fc1(features)
        predicate_feature = self.predicate_ac1(predicate_feature)
        predicate = self.predicate_fc2(predicate_feature)  # [n^2, 249]
        predicate = self.predicate_ac2(predicate)

        predicates = []
        predicate_features = []
        pair_count = 0
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            predicates.append(predicate[pair_count:pair_count + len(pred_pair_instance)])
            predicate_features.append(predicate_feature[pair_count:pair_count + len(pred_pair_instance)])
            pair_count += len(pred_pair_instance)

        print("predicate")
        # print(predicates)
        tps = 0
        ps = 0
        tp20s = 0
        p20s = 0
        tp50s = 0
        p50s = 0
        tp100s = 0
        p100s = 0
        gs = 0
        if training:
            # print(predicate_gts)
            alpha = self.unlabeled_weight(iteration)
            for i in range(len(pred_pair_instances)):
                predicate = predicates[i]
                predicate_gt = predicate_gts[i]
                labeled_pair_location = labeled_pair_locations[i]
                unlabeled_pair_location = unlabeled_pair_locations[i]

                labeled_predicate = predicate[labeled_pair_location]
                labeled_predicate_gt = predicate_gt[labeled_pair_location]
                # print(labeled_predicate.shape)
                unlabeled_predicate = predicate[unlabeled_pair_location]
                topk = int(torch.sum(labeled_predicate_gt.flatten()).item())
                if topk > 0:
                    threk = torch.topk(labeled_predicate.flatten(), topk)[0][-1]
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > threk,
                                                         torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))
                else:
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > 1, torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))

                labeled_predicate_pos_loss, labeled_predicate_neg_loss = self.binary_focal_loss(
                    labeled_predicate.flatten(), labeled_predicate_gt.flatten())

                unlabeled_predicate_pos_loss, unlabeled_predicate_neg_loss = self.binary_focal_loss(
                    unlabeled_predicate.flatten(), unlabeled_predicate_gt.flatten())


                if i == 0:
                    losses['predicate_pos_loss'] = labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                    losses['predicate_neg_loss'] = labeled_predicate_neg_loss + alpha * unlabeled_predicate_neg_loss
                else:
                    losses['predicate_pos_loss'] += labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                    losses['predicate_neg_loss'] += labeled_predicate_neg_loss + alpha * unlabeled_predicate_neg_loss

                predicate_gt = labeled_predicate_gt.flatten()
                predicate_pred = labeled_predicate.flatten()
                predicate_k = int(torch.sum(predicate_gt).item())
                if predicate_k > 0:
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
            metrics['predicate_tp'] = tps
            metrics['predicate_tp20'] = tp20s
            metrics['predicate_tp50'] = tp50s
            metrics['predicate_tp100'] = tp100s
            metrics['predicate_p'] = ps
            metrics['predicate_p20'] = p20s
            metrics['predicate_p50'] = p50s
            metrics['predicate_p100'] = p100s
            metrics['predicate_g'] = gs

        return None, predicates, None, predicate_features, losses, metrics

    def unlabeled_weight(self, iteration):
        alpha = 0.0
        if iteration > self.unlabel_iteration_threshold1:
            alpha = (iteration - self.unlabel_iteration_threshold1) / (
                        self.unlabel_iteration_threshold2 - self.unlabel_iteration_threshold1) * self.af
            if iteration > self.unlabel_iteration_threshold2:
                alpha = self.af
        return alpha

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

@RELATION_PREDICATE_HEAD_REGISTRY.register()
class PredicateHeadsMFULN7(nn.Module):  # 45 without pair
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.use_bias = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.USE_BIAS

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        # self.semantic_embed.weight.requires_grad=False

        self.lstm = nn.LSTM(300, 300, num_layers=1, batch_first=True)
        self.language_lstm_fc = nn.Linear(300, 512)

        self.language_fc = nn.Linear(512, 512)
        self.visual_fc = nn.Linear(2048, 512)
        self.union_location_fc = nn.Linear(8, 512)

        self.predicate_fc1 = nn.Linear(512 * 3, 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num - 1)
        self.predicate_ac2 = nn.Sigmoid()

        if self.use_bias:
            predicate_matrix = np.load("predicate_matrix.npy")
            predicate_matrix /= np.sum(predicate_matrix, 2)[:, :, None]
            self.freq_bias = FrequencyBias(cfg, predicate_matrix)

        self.unlabel_iteration_threshold1 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD1
        self.unlabel_iteration_threshold2 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD2
        self.af = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.AF

        self.predicate_loss_func = nn.BCELoss()

    def forward(self,
                pred_instances, pred_features, instance_interest_preds, relation_instance_features,
                pred_pair_instances, pred_pair_predicate_features, pair_interest_preds, relation_pair_features,
                mannual_triplets, training=True, iteration=1):
        losses = {}
        metrics = {}

        subobj_categories_mix = []
        pred_pair_predicate_features_mix = []
        pred_pair_union_locations_mix = []
        predicate_gts = []
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            sub_classes = pred_pair_instance.pred_pair_sub_classes - 1
            obj_classes = pred_pair_instance.pred_pair_obj_classes - 1
            pred_pair_predicate_features_this = pred_pair_predicate_features[i]
            pred_pair_union_locations_this = pred_pair_instance.pred_pair_union_locations
            subobj_categories = torch.cat([sub_classes.unsqueeze(1), obj_classes.unsqueeze(1)], dim=1).long()
            subobj_categories_mix.append(subobj_categories)
            pred_pair_predicate_features_mix.append(pred_pair_predicate_features_this)
            pred_pair_union_locations_mix.append(pred_pair_union_locations_this)
            if training:
                predicate_gts.append(pred_pair_instance.pred_pair_gt_predicate_full[:, 1:])

        subobj_categories_mix = torch.cat(subobj_categories_mix)
        pred_pair_predicate_features_mix = torch.cat(pred_pair_predicate_features_mix)
        pred_pair_union_locations_mix = torch.cat(pred_pair_union_locations_mix)

        subobj_embeddings = self.semantic_embed(subobj_categories_mix)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(subobj_embeddings)
        language_lstm_out = self.language_lstm_fc(lstm_out[:, -1, :])
        language_out = F.relu(self.language_fc(language_lstm_out))

        visual_out = F.relu(self.visual_fc(pred_pair_predicate_features_mix))
        union_location_out = F.relu(self.union_location_fc(pred_pair_union_locations_mix))

        features = torch.cat([language_out, visual_out, union_location_out], dim=1)

        predicate_feature = self.predicate_fc1(features)
        predicate_feature = self.predicate_ac1(predicate_feature)
        predicate = self.predicate_fc2(predicate_feature)  # [n^2, 249]
        predicate = self.predicate_ac2(predicate)

        predicates = []
        predicate_features = []
        pair_count = 0
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            predicates.append(predicate[pair_count:pair_count + len(pred_pair_instance)])
            predicate_features.append(predicate_feature[pair_count:pair_count + len(pred_pair_instance)])
            pair_count += len(pred_pair_instance)

        print("predicate")
        # print(predicates)
        tps = 0
        ps = 0
        tp20s = 0
        p20s = 0
        tp50s = 0
        p50s = 0
        tp100s = 0
        p100s = 0
        gs = 0
        if training:
            # print(predicate_gts)
            for i in range(len(pred_pair_instances)):
                predicate = predicates[i]
                predicate_gt = predicate_gts[i]

                predicate_pos_loss, predicate_neg_loss = self.binary_focal_loss(predicate.flatten(),predicate_gt.flatten())

                losses['predicate_pos_loss'] = predicate_pos_loss
                losses['predicate_neg_loss'] = predicate_neg_loss

                predicate_gt = predicate_gt.flatten()
                predicate_pred = predicate.flatten()
                predicate_k = int(torch.sum(predicate_gt).item())
                if predicate_k > 0:
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
            metrics['predicate_tp'] = tps
            metrics['predicate_tp20'] = tp20s
            metrics['predicate_tp50'] = tp50s
            metrics['predicate_tp100'] = tp100s
            metrics['predicate_p'] = ps
            metrics['predicate_p20'] = p20s
            metrics['predicate_p50'] = p50s
            metrics['predicate_p100'] = p100s
            metrics['predicate_g'] = gs

        return None, predicates, None, predicate_features, losses, metrics

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

@RELATION_PREDICATE_HEAD_REGISTRY.register()
class PredicateHeadsMFULN8(nn.Module):  # 45 without semantic
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.use_bias = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.USE_BIAS

        self.visual_fc = nn.Linear(2048, 512)
        self.union_location_fc = nn.Linear(8, 512)

        self.predicate_fc1 = nn.Linear(512 * 2, 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num - 1)
        self.predicate_ac2 = nn.Sigmoid()

        if self.use_bias:
            predicate_matrix = np.load("predicate_matrix.npy")
            predicate_matrix /= np.sum(predicate_matrix, 2)[:, :, None]
            self.freq_bias = FrequencyBias(cfg, predicate_matrix)

        self.unlabel_iteration_threshold1 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD1
        self.unlabel_iteration_threshold2 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD2
        self.af = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.AF

        self.predicate_loss_func = nn.BCELoss()

    def forward(self,
                pred_instances, pred_features, instance_interest_preds, relation_instance_features,
                pred_pair_instances, pred_pair_predicate_features, pair_interest_preds, relation_pair_features,
                mannual_triplets, training=True, iteration=1):
        losses = {}
        metrics = {}

        subobj_categories_mix = []
        pred_pair_predicate_features_mix = []
        pred_pair_union_locations_mix = []
        predicate_gts = []
        labeled_pair_locations = []
        unlabeled_pair_locations = []
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            sub_classes = pred_pair_instance.pred_pair_sub_classes - 1
            obj_classes = pred_pair_instance.pred_pair_obj_classes - 1
            pred_pair_predicate_features_this = pred_pair_predicate_features[i]
            pred_pair_union_locations_this = pred_pair_instance.pred_pair_union_locations
            subobj_categories = torch.cat([sub_classes.unsqueeze(1), obj_classes.unsqueeze(1)], dim=1).long()
            subobj_categories_mix.append(subobj_categories)
            pred_pair_predicate_features_mix.append(pred_pair_predicate_features_this)
            pred_pair_union_locations_mix.append(pred_pair_union_locations_this)
            if training:
                labeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest > 0))
                unlabeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest <= 0))
                predicate_gts.append(pred_pair_instance.pred_pair_gt_predicate_full[:, 1:])

        subobj_categories_mix = torch.cat(subobj_categories_mix)
        pred_pair_predicate_features_mix = torch.cat(pred_pair_predicate_features_mix)
        pred_pair_union_locations_mix = torch.cat(pred_pair_union_locations_mix)

        visual_out = F.relu(self.visual_fc(pred_pair_predicate_features_mix))
        union_location_out = F.relu(self.union_location_fc(pred_pair_union_locations_mix))

        features = torch.cat([visual_out, union_location_out], dim=1)

        predicate_feature = self.predicate_fc1(features)
        predicate_feature = self.predicate_ac1(predicate_feature)
        predicate = self.predicate_fc2(predicate_feature)  # [n^2, 249]
        predicate = self.predicate_ac2(predicate)

        predicates = []
        predicate_features = []
        pair_count = 0
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            predicates.append(predicate[pair_count:pair_count + len(pred_pair_instance)])
            predicate_features.append(predicate_feature[pair_count:pair_count + len(pred_pair_instance)])
            pair_count += len(pred_pair_instance)

        print("predicate")
        # print(predicates)
        tps = 0
        ps = 0
        tp20s = 0
        p20s = 0
        tp50s = 0
        p50s = 0
        tp100s = 0
        p100s = 0
        gs = 0
        if training:
            # print(predicate_gts)
            alpha = self.unlabeled_weight(iteration)
            for i in range(len(pred_pair_instances)):
                predicate = predicates[i]
                predicate_gt = predicate_gts[i]
                labeled_pair_location = labeled_pair_locations[i]
                unlabeled_pair_location = unlabeled_pair_locations[i]

                labeled_predicate = predicate[labeled_pair_location]
                labeled_predicate_gt = predicate_gt[labeled_pair_location]
                # print(labeled_predicate.shape)
                unlabeled_predicate = predicate[unlabeled_pair_location]
                topk = int(torch.sum(labeled_predicate_gt.flatten()).item())
                if topk > 0:
                    threk = torch.topk(labeled_predicate.flatten(), topk)[0][-1]
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > threk,
                                                         torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))
                else:
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > 1, torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))

                labeled_predicate_pos_loss, labeled_predicate_neg_loss = self.binary_focal_loss(
                    labeled_predicate.flatten(), labeled_predicate_gt.flatten())

                unlabeled_predicate_pos_loss, unlabeled_predicate_neg_loss = self.binary_focal_loss(
                    unlabeled_predicate.flatten(), unlabeled_predicate_gt.flatten())


                if i == 0:
                    losses['predicate_pos_loss'] = labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                    losses['predicate_neg_loss'] = labeled_predicate_neg_loss + alpha * unlabeled_predicate_neg_loss
                else:
                    losses['predicate_pos_loss'] += labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                    losses['predicate_neg_loss'] += labeled_predicate_neg_loss + alpha * unlabeled_predicate_neg_loss

                predicate_gt = labeled_predicate_gt.flatten()
                predicate_pred = labeled_predicate.flatten()
                predicate_k = int(torch.sum(predicate_gt).item())
                if predicate_k > 0:
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
            metrics['predicate_tp'] = tps
            metrics['predicate_tp20'] = tp20s
            metrics['predicate_tp50'] = tp50s
            metrics['predicate_tp100'] = tp100s
            metrics['predicate_p'] = ps
            metrics['predicate_p20'] = p20s
            metrics['predicate_p50'] = p50s
            metrics['predicate_p100'] = p100s
            metrics['predicate_g'] = gs

        return None, predicates, None, predicate_features, losses, metrics

    def unlabeled_weight(self, iteration):
        alpha = 0.0
        if iteration > self.unlabel_iteration_threshold1:
            alpha = (iteration - self.unlabel_iteration_threshold1) / (
                        self.unlabel_iteration_threshold2 - self.unlabel_iteration_threshold1) * self.af
            if iteration > self.unlabel_iteration_threshold2:
                alpha = self.af
        return alpha

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

@RELATION_PREDICATE_HEAD_REGISTRY.register()
class PredicateHeadsMFULN9(nn.Module):  # 45 without location
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.use_bias = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.USE_BIAS

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        # self.semantic_embed.weight.requires_grad=False

        self.lstm = nn.LSTM(300, 300, num_layers=1, batch_first=True)
        self.language_lstm_fc = nn.Linear(300, 512)

        self.language_fc = nn.Linear(512, 512)
        self.visual_fc = nn.Linear(2048, 512)

        self.predicate_fc1 = nn.Linear(512 * 2, 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num - 1)
        self.predicate_ac2 = nn.Sigmoid()

        if self.use_bias:
            predicate_matrix = np.load("predicate_matrix.npy")
            predicate_matrix /= np.sum(predicate_matrix, 2)[:, :, None]
            self.freq_bias = FrequencyBias(cfg, predicate_matrix)

        self.unlabel_iteration_threshold1 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD1
        self.unlabel_iteration_threshold2 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD2
        self.af = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.AF

        self.predicate_loss_func = nn.BCELoss()

    def forward(self,
                pred_instances, pred_features, instance_interest_preds, relation_instance_features,
                pred_pair_instances, pred_pair_predicate_features, pair_interest_preds, relation_pair_features,
                mannual_triplets, training=True, iteration=1):
        losses = {}
        metrics = {}

        subobj_categories_mix = []
        pred_pair_predicate_features_mix = []
        pred_pair_union_locations_mix = []
        predicate_gts = []
        labeled_pair_locations = []
        unlabeled_pair_locations = []
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            sub_classes = pred_pair_instance.pred_pair_sub_classes - 1
            obj_classes = pred_pair_instance.pred_pair_obj_classes - 1
            pred_pair_predicate_features_this = pred_pair_predicate_features[i]
            pred_pair_union_locations_this = pred_pair_instance.pred_pair_union_locations
            subobj_categories = torch.cat([sub_classes.unsqueeze(1), obj_classes.unsqueeze(1)], dim=1).long()
            subobj_categories_mix.append(subobj_categories)
            pred_pair_predicate_features_mix.append(pred_pair_predicate_features_this)
            pred_pair_union_locations_mix.append(pred_pair_union_locations_this)
            if training:
                labeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest > 0))
                unlabeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest <= 0))
                predicate_gts.append(pred_pair_instance.pred_pair_gt_predicate_full[:, 1:])

        subobj_categories_mix = torch.cat(subobj_categories_mix)
        pred_pair_predicate_features_mix = torch.cat(pred_pair_predicate_features_mix)
        pred_pair_union_locations_mix = torch.cat(pred_pair_union_locations_mix)

        subobj_embeddings = self.semantic_embed(subobj_categories_mix)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(subobj_embeddings)
        language_lstm_out = self.language_lstm_fc(lstm_out[:, -1, :])
        language_out = F.relu(self.language_fc(language_lstm_out))

        visual_out = F.relu(self.visual_fc(pred_pair_predicate_features_mix))

        features = torch.cat([language_out, visual_out], dim=1)

        predicate_feature = self.predicate_fc1(features)
        predicate_feature = self.predicate_ac1(predicate_feature)
        predicate = self.predicate_fc2(predicate_feature)  # [n^2, 249]
        predicate = self.predicate_ac2(predicate)

        predicates = []
        predicate_features = []
        pair_count = 0
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            predicates.append(predicate[pair_count:pair_count + len(pred_pair_instance)])
            predicate_features.append(predicate_feature[pair_count:pair_count + len(pred_pair_instance)])
            pair_count += len(pred_pair_instance)

        print("predicate")
        # print(predicates)
        tps = 0
        ps = 0
        tp20s = 0
        p20s = 0
        tp50s = 0
        p50s = 0
        tp100s = 0
        p100s = 0
        gs = 0
        if training:
            # print(predicate_gts)
            alpha = self.unlabeled_weight(iteration)
            for i in range(len(pred_pair_instances)):
                predicate = predicates[i]
                predicate_gt = predicate_gts[i]
                labeled_pair_location = labeled_pair_locations[i]
                unlabeled_pair_location = unlabeled_pair_locations[i]

                labeled_predicate = predicate[labeled_pair_location]
                labeled_predicate_gt = predicate_gt[labeled_pair_location]
                # print(labeled_predicate.shape)
                unlabeled_predicate = predicate[unlabeled_pair_location]
                topk = int(torch.sum(labeled_predicate_gt.flatten()).item())
                if topk > 0:
                    threk = torch.topk(labeled_predicate.flatten(), topk)[0][-1]
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > threk,
                                                         torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))
                else:
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > 1, torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))

                labeled_predicate_pos_loss, labeled_predicate_neg_loss = self.binary_focal_loss(
                    labeled_predicate.flatten(), labeled_predicate_gt.flatten())

                unlabeled_predicate_pos_loss, unlabeled_predicate_neg_loss = self.binary_focal_loss(
                    unlabeled_predicate.flatten(), unlabeled_predicate_gt.flatten())


                if i == 0:
                    losses['predicate_pos_loss'] = labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                    losses['predicate_neg_loss'] = labeled_predicate_neg_loss + alpha * unlabeled_predicate_neg_loss
                else:
                    losses['predicate_pos_loss'] += labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                    losses['predicate_neg_loss'] += labeled_predicate_neg_loss + alpha * unlabeled_predicate_neg_loss

                predicate_gt = labeled_predicate_gt.flatten()
                predicate_pred = labeled_predicate.flatten()
                predicate_k = int(torch.sum(predicate_gt).item())
                if predicate_k > 0:
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
            metrics['predicate_tp'] = tps
            metrics['predicate_tp20'] = tp20s
            metrics['predicate_tp50'] = tp50s
            metrics['predicate_tp100'] = tp100s
            metrics['predicate_p'] = ps
            metrics['predicate_p20'] = p20s
            metrics['predicate_p50'] = p50s
            metrics['predicate_p100'] = p100s
            metrics['predicate_g'] = gs

        return None, predicates, None, predicate_features, losses, metrics

    def unlabeled_weight(self, iteration):
        alpha = 0.0
        if iteration > self.unlabel_iteration_threshold1:
            alpha = (iteration - self.unlabel_iteration_threshold1) / (
                        self.unlabel_iteration_threshold2 - self.unlabel_iteration_threshold1) * self.af
            if iteration > self.unlabel_iteration_threshold2:
                alpha = self.af
        return alpha

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

@RELATION_PREDICATE_HEAD_REGISTRY.register()
class PredicateHeadsMFULN10(nn.Module):  # 45 with bceloss
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.use_bias = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.USE_BIAS

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        # self.semantic_embed.weight.requires_grad=False

        self.lstm = nn.LSTM(300, 300, num_layers=1, batch_first=True)
        self.language_lstm_fc = nn.Linear(300, 512)

        self.language_fc = nn.Linear(512, 512)
        self.visual_fc = nn.Linear(2048, 512)
        self.union_location_fc = nn.Linear(8, 512)

        self.predicate_fc1 = nn.Linear(512 * 3, 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num - 1)
        self.predicate_ac2 = nn.Sigmoid()

        if self.use_bias:
            predicate_matrix = np.load("predicate_matrix.npy")
            predicate_matrix /= np.sum(predicate_matrix, 2)[:, :, None]
            self.freq_bias = FrequencyBias(cfg, predicate_matrix)

        self.unlabel_iteration_threshold1 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD1
        self.unlabel_iteration_threshold2 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD2
        self.af = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.AF

        self.predicate_loss_func = nn.BCELoss()

    def forward(self,
                pred_instances, pred_features, instance_interest_preds, relation_instance_features,
                pred_pair_instances, pred_pair_predicate_features, pair_interest_preds, relation_pair_features,
                mannual_triplets, training=True, iteration=1):
        losses = {}
        metrics = {}

        subobj_categories_mix = []
        pred_pair_predicate_features_mix = []
        pred_pair_union_locations_mix = []
        predicate_gts = []
        labeled_pair_locations = []
        unlabeled_pair_locations = []
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            sub_classes = pred_pair_instance.pred_pair_sub_classes - 1
            obj_classes = pred_pair_instance.pred_pair_obj_classes - 1
            pred_pair_predicate_features_this = pred_pair_predicate_features[i]
            pred_pair_union_locations_this = pred_pair_instance.pred_pair_union_locations
            subobj_categories = torch.cat([sub_classes.unsqueeze(1), obj_classes.unsqueeze(1)], dim=1).long()
            subobj_categories_mix.append(subobj_categories)
            pred_pair_predicate_features_mix.append(pred_pair_predicate_features_this)
            pred_pair_union_locations_mix.append(pred_pair_union_locations_this)
            if training:
                labeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest > 0))
                unlabeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest <= 0))
                predicate_gts.append(pred_pair_instance.pred_pair_gt_predicate_full[:, 1:])

        subobj_categories_mix = torch.cat(subobj_categories_mix)
        pred_pair_predicate_features_mix = torch.cat(pred_pair_predicate_features_mix)
        pred_pair_union_locations_mix = torch.cat(pred_pair_union_locations_mix)

        subobj_embeddings = self.semantic_embed(subobj_categories_mix)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(subobj_embeddings)
        language_lstm_out = self.language_lstm_fc(lstm_out[:, -1, :])
        language_out = F.relu(self.language_fc(language_lstm_out))

        visual_out = F.relu(self.visual_fc(pred_pair_predicate_features_mix))
        union_location_out = F.relu(self.union_location_fc(pred_pair_union_locations_mix))

        features = torch.cat([language_out, visual_out, union_location_out], dim=1)

        predicate_feature = self.predicate_fc1(features)
        predicate_feature = self.predicate_ac1(predicate_feature)
        predicate = self.predicate_fc2(predicate_feature)  # [n^2, 249]
        predicate = self.predicate_ac2(predicate)

        predicates = []
        predicate_features = []
        pair_count = 0
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            predicates.append(predicate[pair_count:pair_count + len(pred_pair_instance)])
            predicate_features.append(predicate_feature[pair_count:pair_count + len(pred_pair_instance)])
            pair_count += len(pred_pair_instance)

        print("predicate")
        # print(predicates)
        tps = 0
        ps = 0
        tp20s = 0
        p20s = 0
        tp50s = 0
        p50s = 0
        tp100s = 0
        p100s = 0
        gs = 0
        if training:
            # print(predicate_gts)
            alpha = self.unlabeled_weight(iteration)
            for i in range(len(pred_pair_instances)):
                predicate = predicates[i]
                predicate_gt = predicate_gts[i]
                labeled_pair_location = labeled_pair_locations[i]
                unlabeled_pair_location = unlabeled_pair_locations[i]

                labeled_predicate = predicate[labeled_pair_location]
                labeled_predicate_gt = predicate_gt[labeled_pair_location]
                # print(labeled_predicate.shape)
                unlabeled_predicate = predicate[unlabeled_pair_location]
                topk = int(torch.sum(labeled_predicate_gt.flatten()).item())
                if topk > 0:
                    threk = torch.topk(labeled_predicate.flatten(), topk)[0][-1]
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > threk,
                                                         torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))
                else:
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > 1, torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))

                labeled_predicate_pos_loss = F.binary_cross_entropy(labeled_predicate.flatten(), labeled_predicate_gt.flatten())

                unlabeled_predicate_pos_loss = F.binary_cross_entropy(
                    unlabeled_predicate.flatten(), unlabeled_predicate_gt.flatten())

                if i == 0:
                    losses['predicate_pos_loss'] = labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                else:
                    losses['predicate_pos_loss'] += labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss

                predicate_gt = labeled_predicate_gt.flatten()
                predicate_pred = labeled_predicate.flatten()
                predicate_k = int(torch.sum(predicate_gt).item())
                if predicate_k > 0:
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
            metrics['predicate_tp'] = tps
            metrics['predicate_tp20'] = tp20s
            metrics['predicate_tp50'] = tp50s
            metrics['predicate_tp100'] = tp100s
            metrics['predicate_p'] = ps
            metrics['predicate_p20'] = p20s
            metrics['predicate_p50'] = p50s
            metrics['predicate_p100'] = p100s
            metrics['predicate_g'] = gs

        return None, predicates, None, predicate_features, losses, metrics

    def unlabeled_weight(self, iteration):
        alpha = 0.0
        if iteration > self.unlabel_iteration_threshold1:
            alpha = (iteration - self.unlabel_iteration_threshold1) / (
                        self.unlabel_iteration_threshold2 - self.unlabel_iteration_threshold1) * self.af
            if iteration > self.unlabel_iteration_threshold2:
                alpha = self.af
        return alpha

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

class FrequencyBias(nn.Module):
    def __init__(self,cfg,pred_dist):
        super(FrequencyBias, self).__init__()
        self.device=cfg.MODEL.DEVICE

        self.num_objs=pred_dist.shape[0] # 133
        pred_dist=torch.FloatTensor(pred_dist).view(-1,pred_dist.shape[2]).to(self.device) # 133^2,249

        self.obj_baseline = nn.Embedding(pred_dist.size(0),pred_dist.size(1))
        self.obj_baseline.weight.data=pred_dist

    def index_with_labels(self,sub_classes,obj_classes):
        return self.obj_baseline(sub_classes*self.num_objs+obj_classes)

    def forward(self,obj_cands0,obj_cands1):
        joint_cands=obj_cands0[:,:,None]*obj_cands[:,None]
        baseline=joint_cands.view(joint_cands.size(0),-1)@self.obj_baseline.weight
        return baseline

def compute_predicate_result(pred_confidence, pred_predicate, predicate_confidences,predicate_categories_onehots,
                             predicate_feature, instance_nums, losses, metrics, loss_func,training):
    confidences = []
    predicates=[]
    predicate_features = []
    pair_count = 0

    for instance_num in instance_nums:
        if pred_confidence is not None:
            confidences.append(pred_confidence[pair_count:pair_count + instance_num])
        predicates.append(pred_predicate[pair_count:pair_count + instance_num])
        predicate_features.append(predicate_feature[pair_count:pair_count + instance_num])
        pair_count += instance_num

    if pred_confidence is not None:
        print("confidence")
        # print(confidences)
        if training:
            predicate_confidence_mix=torch.cat(predicate_confidences)
            losses['confidence_loss'] = loss_func(pred_confidence, predicate_confidence_mix)
            # print(predicate_confidences)

            tps = 0
            ps = 0
            gs = 0
            for i in range(len(instance_nums)):
                confidence=confidences[i]
                predicate_confidence=predicate_confidences[i]
                confidence_k = int(torch.sum(predicate_confidence).item())
                if confidence_k>predicate_confidence.shape[0]:
                    confidence_k=predicate_confidence.shape[0]
                    print(confidence_k)
                if confidence_k > 0:
                    confidence_pred_score, confidence_pred_index = torch.topk(confidence, confidence_k)
                    print(confidence_pred_score)
                    print(str(confidence_k) + " " + str(torch.sum(confidence >= confidence_pred_score[-1].item()).item()))
                    confidence_pred_pred = torch.where(confidence >= confidence_pred_score[-1].item(),
                                                       torch.ones_like(confidence), torch.zeros_like(confidence))
                    tp = torch.sum(confidence_pred_pred * predicate_confidence)
                    p = torch.sum(confidence_pred_pred)
                else:
                    tp = 0
                    p = 0
                g = torch.sum(predicate_confidence)
                tps+=tp
                ps+=p
                gs+=g
                # print("instance-tp:" + str(tp) + " p:" + str(p) + " g:" + str(g))
                metrics['confidence_tp'] = tps
                metrics['confidence_p'] = ps
                metrics['confidence_g'] = gs

    print("predicate")
    # print(predicates)
    if training:
        predicate_categories_onehot_mix = torch.cat(predicate_categories_onehots)
        losses['predicate_loss'] = loss_func(pred_predicate.flatten(), predicate_categories_onehot_mix.flatten())
        # print(predicate_categories_onehots)
        tps=0
        tp20s=0
        tp50s=0
        tp100s=0
        ps=0
        p20s=0
        p50s=0
        p100s=0
        gs=0
        for i in range(len(instance_nums)):
            predicate_gt = predicate_categories_onehots[i].flatten()
            predicate_pred = predicates[i].flatten()
            if predicate_pred.shape[0]==0:
                tp = 0
                tp20 = 0
                tp50 = 0
                tp100 = 0
                p=0
                p20 = 0
                p50 = 0
                p100 = 0
            else:
                predicate_pred_score20, confidence_pred_index20 = torch.topk(predicate_pred, 20)
                predicate_pred_pred20 = torch.where(predicate_pred >= predicate_pred_score20[-1].item(),
                                                    torch.ones_like(predicate_pred), torch.zeros_like(predicate_pred))
                tp20 = torch.sum(predicate_pred_pred20 * predicate_gt)
                p20 = torch.sum(predicate_pred_pred20)
                predicate_pred_score50, confidence_pred_index50 = torch.topk(predicate_pred, 50)
                predicate_pred_pred50 = torch.where(predicate_pred >= predicate_pred_score50[-1].item(),
                                                    torch.ones_like(predicate_pred), torch.zeros_like(predicate_pred))
                tp50 = torch.sum(predicate_pred_pred50 * predicate_gt)
                p50 = torch.sum(predicate_pred_pred50)
                predicate_pred_score100, confidence_pred_index100 = torch.topk(predicate_pred, 100)
                predicate_pred_pred100 = torch.where(predicate_pred >= predicate_pred_score100[-1].item(),
                                                     torch.ones_like(predicate_pred), torch.zeros_like(predicate_pred))
                tp100 = torch.sum(predicate_pred_pred100 * predicate_gt)
                p100 = torch.sum(predicate_pred_pred100)
                predicate_k = int(torch.sum(predicate_gt).item())
                if predicate_k > 0:
                    predicate_pred_score, confidence_pred_index = torch.topk(predicate_pred, predicate_k)
                    print(predicate_pred_score)
                    print(str(predicate_k) + " " + str(torch.sum(predicate_pred >= predicate_pred_score[-1].item()).item()))
                    predicate_pred_pred = torch.where(predicate_pred >= predicate_pred_score[-1].item(),
                                                      torch.ones_like(predicate_pred), torch.zeros_like(predicate_pred))
                    tp = torch.sum(predicate_pred_pred * predicate_gt)
                    p = torch.sum(predicate_pred_pred)
                else:
                    tp = 0
                    p = 0
            g = torch.sum(predicate_gt)
            tps+=tp
            tp20s+=tp20
            tp50s+=tp50
            tp100s+=tp100
            ps+=p
            p20s+=p20
            p50s+=p50
            p100s+=p100
            gs+=g
            # print("instance-tp:" + str(tp) + " p:" + str(p) + " g:" + str(g))
        metrics['predicate_tp'] = tps
        metrics['predicate_tp20'] = tp20s
        metrics['predicate_tp50'] = tp50s
        metrics['predicate_tp100'] = tp100s
        metrics['predicate_p'] = ps
        metrics['predicate_p20'] = p20s
        metrics['predicate_p50'] = p50s
        metrics['predicate_p100'] = p100s
        metrics['predicate_g'] = gs
    return confidences, predicates, predicate_features, losses, metrics

@RELATION_PREDICATE_HEAD_REGISTRY.register()
class PredicateHeadsMFULN_MEANTEACHER1(nn.Module):  # 4 with semi2
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.instance_num = cfg.MODEL.RELATION_HEADS.INSTANCE_NUM
        self.use_bias = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.USE_BIAS

        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)
        self.semantic_embed = nn.Embedding(self.instance_num - 1, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        # self.semantic_embed.weight.requires_grad=False

        self.lstm = nn.LSTM(300, 300, num_layers=1, batch_first=True)
        self.language_lstm_fc = nn.Linear(300, 512)

        self.language_fc = nn.Linear(512, 512)
        self.visual_fc = nn.Linear(2048, 512)
        self.union_location_fc = nn.Linear(8, 512)

        self.predicate_fc1 = nn.Linear(512 * 3, 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num - 1)
        self.predicate_ac2 = nn.Sigmoid()

        if self.use_bias:
            predicate_matrix = np.load("predicate_matrix.npy")
            predicate_matrix /= np.sum(predicate_matrix, 2)[:, :, None]
            self.freq_bias = FrequencyBias(cfg, predicate_matrix)

        self.unlabel_iteration_threshold1 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD1
        self.unlabel_iteration_threshold2 = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.UNLABELED_ITERATION_THRESHOLD2
        self.af = cfg.MODEL.RELATION_HEADS.RELATION_PREDICATE_HEAD.AF

        self.predicate_loss_func = nn.BCELoss()

    def forward(self,
                pred_instances, pred_features, instance_interest_preds, relation_instance_features,
                pred_pair_instances, pred_pair_predicate_features, pair_interest_preds, relation_pair_features,
                mannual_triplets, training=True, iteration=1):
        losses = {}
        metrics = {}

        subobj_categories_mix = []
        pred_pair_predicate_features_mix = []
        pred_pair_union_locations_mix = []
        predicate_gts = []
        labeled_pair_locations = []
        unlabeled_pair_locations = []
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            sub_classes = pred_pair_instance.pred_pair_sub_classes - 1
            obj_classes = pred_pair_instance.pred_pair_obj_classes - 1
            pred_pair_predicate_features_this = pred_pair_predicate_features[i]
            pred_pair_union_locations_this = pred_pair_instance.pred_pair_union_locations
            subobj_categories = torch.cat([sub_classes.unsqueeze(1), obj_classes.unsqueeze(1)], dim=1).long()
            subobj_categories_mix.append(subobj_categories)
            pred_pair_predicate_features_mix.append(pred_pair_predicate_features_this)
            pred_pair_union_locations_mix.append(pred_pair_union_locations_this)
            if training:
                labeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest > 0))
                unlabeled_pair_locations.append(torch.where(pred_pair_instance.pred_pair_interest <= 0))
                predicate_gts.append(pred_pair_instance.pred_pair_gt_predicate_full[:, 1:])

        subobj_categories_mix = torch.cat(subobj_categories_mix)
        pred_pair_predicate_features_mix = torch.cat(pred_pair_predicate_features_mix)
        pred_pair_union_locations_mix = torch.cat(pred_pair_union_locations_mix)

        subobj_embeddings = self.semantic_embed(subobj_categories_mix)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(subobj_embeddings)
        language_lstm_out = self.language_lstm_fc(lstm_out[:, -1, :])
        language_out = F.relu(self.language_fc(language_lstm_out))

        visual_out = F.relu(self.visual_fc(pred_pair_predicate_features_mix))
        union_location_out = F.relu(self.union_location_fc(pred_pair_union_locations_mix))

        features = torch.cat([language_out, visual_out, union_location_out], dim=1)

        predicate_feature = self.predicate_fc1(features)
        predicate_feature = self.predicate_ac1(predicate_feature)
        predicate = self.predicate_fc2(predicate_feature)  # [n^2, 249]
        predicate = self.predicate_ac2(predicate)

        predicates = []
        predicate_features = []
        pair_count = 0
        for i in range(len(pred_pair_instances)):
            pred_pair_instance = pred_pair_instances[i]
            predicates.append(predicate[pair_count:pair_count + len(pred_pair_instance)])
            predicate_features.append(predicate_feature[pair_count:pair_count + len(pred_pair_instance)])
            pair_count += len(pred_pair_instance)

        print("predicate")
        # print(predicates)
        tps = 0
        ps = 0
        tp20s = 0
        p20s = 0
        tp50s = 0
        p50s = 0
        tp100s = 0
        p100s = 0
        gs = 0
        if training:
            # print(predicate_gts)
            alpha = self.unlabeled_weight(iteration)
            for i in range(len(pred_pair_instances)):
                predicate = predicates[i]
                predicate_gt = predicate_gts[i]
                labeled_pair_location = labeled_pair_locations[i]
                unlabeled_pair_location = unlabeled_pair_locations[i]

                labeled_predicate = predicate[labeled_pair_location]
                labeled_predicate_gt = predicate_gt[labeled_pair_location]
                # print(labeled_predicate.shape)
                unlabeled_predicate = predicate[unlabeled_pair_location]
                topk = int(torch.sum(labeled_predicate_gt.flatten()).item())
                if topk > 0:
                    threk = torch.topk(labeled_predicate.flatten(), topk)[0][-1]
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > threk,
                                                         torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))
                else:
                    unlabeled_predicate_gt = torch.where(unlabeled_predicate > 1, torch.ones_like(unlabeled_predicate),
                                                         torch.zeros_like(unlabeled_predicate))

                labeled_predicate_pos_loss, labeled_predicate_neg_loss = self.binary_focal_loss(
                    labeled_predicate.flatten(), labeled_predicate_gt.flatten())

                unlabeled_predicate_pos_loss, unlabeled_predicate_neg_loss = self.binary_focal_loss(
                    unlabeled_predicate.flatten(), unlabeled_predicate_gt.flatten())


                if i == 0:
                    losses['predicate_pos_loss'] = labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                    losses['predicate_neg_loss'] = labeled_predicate_neg_loss + alpha * unlabeled_predicate_neg_loss
                else:
                    losses['predicate_pos_loss'] += labeled_predicate_pos_loss + alpha * unlabeled_predicate_pos_loss
                    losses['predicate_neg_loss'] += labeled_predicate_neg_loss + alpha * unlabeled_predicate_neg_loss

                predicate_gt = labeled_predicate_gt.flatten()
                predicate_pred = labeled_predicate.flatten()
                predicate_k = int(torch.sum(predicate_gt).item())
                if predicate_k > 0:
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
            metrics['predicate_tp'] = tps
            metrics['predicate_tp20'] = tp20s
            metrics['predicate_tp50'] = tp50s
            metrics['predicate_tp100'] = tp100s
            metrics['predicate_p'] = ps
            metrics['predicate_p20'] = p20s
            metrics['predicate_p50'] = p50s
            metrics['predicate_p100'] = p100s
            metrics['predicate_g'] = gs

        return None, predicates, None, predicate_features, losses, metrics

    def unlabeled_weight(self, iteration):
        alpha = 0.0
        if iteration > self.unlabel_iteration_threshold1:
            alpha = (iteration - self.unlabel_iteration_threshold1) / (
                        self.unlabel_iteration_threshold2 - self.unlabel_iteration_threshold1) * self.af
            if iteration > self.unlabel_iteration_threshold2:
                alpha = self.af
        return alpha

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