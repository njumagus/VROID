# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch import nn
from torchvision.transforms import Resize
from detectron2.structures import ImageList, Boxes, Instances
from detectron2.utils.torch_utils import extract_bbox
from detectron2.data import MetadataCatalog

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess, sem_seg_postprocess
from ..middleprocessing import generate_thing_instances, generate_stuff_instances,\
    generate_instances,generate_gt_instances,generate_pair_instances,generate_mannual_relation,\
    map_gt_and_relations,generate_instances_interest,generate_pairs_interest
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..relation_heads import build_relation_heads, build_instance_encoder
from .build import META_ARCH_REGISTRY
from .semantic_seg import build_sem_seg_head

__all__ = ["PanopticRelation"]


@META_ARCH_REGISTRY.register()
class PanopticRelation(nn.Module):
    """
    Main class for Panoptic FPN architectures (see https://arxiv.org/abs/1901.02446).
    """

    def __init__(self, cfg):
        super().__init__()

        self.square_image_size=cfg.MODEL.RELATION_HEADS.IMAGE_SIZE

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.instance_loss_weight = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
        self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = (
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH
        )

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.relation_heads = build_relation_heads(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.thing_id_map = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_contiguous_id_to_class_id")
        self.stuff_id_map = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("stuff_contiguous_id_to_class_id")

        if self.thing_id_map is None:
            self.thing_id_map = {i: i + 1 for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
        if self.stuff_id_map is None:
            self.stuff_id_map = {i + 1: i + cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1 for i in range(cfg.MODEL.RELATION_HEADS.INSTANCE_NUM-cfg.MODEL.ROI_HEADS.NUM_CLASSES-1)}
            self.stuff_id_map[0]=0

        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.relation_head_list=cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST
        self.instance_encoder=build_instance_encoder(cfg)

        self.to(self.device)

    def forward(self, batched_inputs,iteration,mode="panoptic",training=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                  See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        if mode=="panoptic":
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            features = self.backbone(images.tensor)

            if "proposals" in batched_inputs[0]:
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            if "sem_seg" in batched_inputs[0]:
                gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
                gt_sem_seg = ImageList.from_tensors(
                    gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
                ).tensor
            else:
                gt_sem_seg = None
            sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            detector_results, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances
            )

            if self.training:
                losses = {}
                losses.update(sem_seg_losses)
                losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
                losses.update(proposal_losses)
                return losses

            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

                if self.combine_on:
                    panoptic_r = combine_semantic_and_instance_outputs(
                        self.thing_id_map,
                        self.stuff_id_map,
                        detector_r,
                        sem_seg_r.argmax(dim=0),
                        self.combine_overlap_threshold,
                        self.combine_stuff_area_limit,
                        self.combine_instances_confidence_threshold,
                        mode=mode
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        elif mode=="relation":
            losses={}
            metrics={}
            # print("origin image: ("+str(batched_inputs[0]['height'])+", "+str(batched_inputs[0]['width'])+")")
            images = [x["image"].to(self.device) for x in batched_inputs]
            # print("input image: "+str(images[0].shape))
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            features = self.backbone(images.tensor)

            image_features = self.roi_heads.generate_instance_box_features(features, [Boxes(torch.IntTensor([[0, 0, image_size[1], image_size[0]]]).to(self.device)) for image_size in images.image_sizes])
            image_features=[image_feature for image_feature in image_features]
            if training:
                gt_thing_instances = [batched_input['instances'].to(self.device) for batched_input in batched_inputs]
                gt_stuff_instances = [batched_input['stuffs'].to(self.device) for batched_input in batched_inputs]
                gt_triplets = [batched_input['triplets'].to(self.device) for batched_input in batched_inputs]

                gt_instances = generate_gt_instances(gt_thing_instances,gt_stuff_instances)
                gt_instances = generate_instances_interest(gt_instances, gt_triplets,self.relation_num)
                gt_instance_nums=[len(gt_instance) for gt_instance in gt_instances]
                gt_box_features_mix = self.roi_heads.generate_instance_box_features(features, [gt_instance.pred_boxes for gt_instance in gt_instances])
                gt_box_features = gt_box_features_mix.split(gt_instance_nums)
                # print(gt_instances)

                # gt_pair_instances = None
                # # gt_pair_box_features = None
                # if "pair" in self.relation_head_list or "predicate" in self.relation_head_list:
                #     gt_pair_instances = generate_pair_instances(gt_instances)
                #     gt_pair_instances = generate_pairs_interest(gt_instances,gt_pair_instances,gt_triplets,self.relation_num)
                #     # gt_pair_box_features = self.roi_heads.generate_instance_box_features(features, [gt_pair_instances.pred_pair_boxes])
                # # print(gt_pair_instances)

                # gt_mannual_triplets=None
                # if "predicate" in self.relation_head_list:
                #     gt_mannual_triplets = generate_mannual_relation(gt_instances,gt_triplets)

                # gt_instance_features, gt_instance_logits, gt_class_loss, gt_class_metirc = self.instance_encoder(
                #     image_feature, None, gt_box_features, gt_instances.pred_classes, training)
                # gt_instances.pred_class_logits = gt_instance_logits
                # losses.update(gt_class_loss)
                # metrics.update(gt_class_metirc)
                #
                # gt_pair_instance_features=None
                # if "pair" in self.relation_head_list or "predicate" in self.relation_head_list:
                #     gt_pair_instance_features, _, _, _ = self.instance_encoder(
                #         image_feature, None, gt_pair_box_features, None, training=False)
                #
                # gt_pair_predicate_features=None
                # if "predicate" in self.relation_head_list:
                #     gt_pair_predicate_features = []
                #     for i in range(len(gt_instances)):
                #         for j in range(len(gt_instances)):
                #             gt_pair_predicate_features.append(torch.cat([gt_instance_features[i] - gt_instance_features[j],gt_pair_instance_features[i * gt_instance_features.shape[0] + j]]))
                #     gt_pair_predicate_features = torch.stack(gt_pair_predicate_features)

                # relation_results, relation_losses, relation_metrics = self.relation_heads(gt_instances,gt_pair_instances,
                #                                                                           gt_instance_features,gt_pair_instance_features,gt_pair_predicate_features,
                #                                                                           gt_mannual_triplets, training,iteration)

            if "proposals" in batched_inputs[0]:
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            if "sem_seg" in batched_inputs[0]:
                gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
                gt_sem_seg = ImageList.from_tensors(gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value).tensor
                gt_stuff_instances = batched_inputs[0]['stuffs'].to(self.device)
            else:
                gt_sem_seg = None
                gt_stuff_instances=None
            sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg, mode=mode)  # [1, 54, 672, 1024]
            stuff_instances = generate_stuff_instances(self.stuff_id_map, sem_seg_results, images.image_sizes)
            stuff_instance_nums=[len(stuff_instance) for stuff_instance in stuff_instances]
            stuff_box_features_mix = self.roi_heads.generate_instance_box_features(features, [stuff_instance.pred_boxes for stuff_instance in stuff_instances])
            stuff_box_features=stuff_box_features_mix.split(stuff_instance_nums)

            if "instances" in batched_inputs[0]:
                gt_t_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_thing_instances = batched_inputs[0]['instances'].to(self.device)
            else:
                gt_t_instances = None
                gt_thing_instances = None
            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_t_instances) # rpn
            detector_results, thing_box_features = self.roi_heads(images, features, proposals, gt_t_instances,mode="relation")
            thing_instances, keep_areas = generate_thing_instances(self.thing_id_map,detector_results)

            # pred_classes,pred_masks,pred_boxes,relation_pred_gt,instance_interest_pred_gt=make_relation_target(batched_inputs[0]['file_name'],self.square_image_size,panoptic_ins,(batched_inputs[0]['height'],batched_inputs[0]['width']),batched_inputs[0]['instances'],batched_inputs[0]['stuffs'],batched_inputs[0]['triplets'])
            pred_instances = generate_instances(thing_instances, stuff_instances)
            print()
            print(thing_instances[0])
            print(stuff_instances[0])
            print(pred_instances[0])
            pred_instance_nums=[len(pred_instance) for pred_instance in pred_instances]
            pred_box_features = []
            for i in range(len(pred_instance_nums)):
                keep_area=keep_areas[i]
                thing_box_feature=thing_box_features[i]
                stuff_box_feature=stuff_box_features[i]
                if len(keep_area)>0 and stuff_box_feature.shape[0]>0:
                    pred_box_feature = torch.cat([thing_box_feature[keep_area[0]], stuff_box_feature])
                elif stuff_box_feature.shape[0]>0:
                    pred_box_feature = stuff_box_feature
                elif len(keep_area)>0:
                    pred_box_feature = thing_box_feature[keep_area[0]]
                else:
                    pred_box_feature=torch.Tensor().to(self.device)
                pred_box_features.append(pred_box_feature)
            # pred_instance_masks=[]
            # image_before_mask=images.tensor[0].permute(1,2,0)[0:images.image_sizes[0][0],0:images.image_sizes[0][1],:]
            # for i in range(pred_instances.pred_masks.shape[0]):
            #     pred_instance_mask=image_before_mask*pred_instances.pred_masks[i].unsqueeze(2)
            #     pred_instance_masks.append(pred_instance_mask)
            # pred_instance_masks=torch.stack(pred_instance_masks).permute(0,3,1,2)
            # print(pred_instance_masks.shape)
            # instance_features=self.backbone(pred_instance_masks)
            # for feature in instance_features:
            #     print(instance_features[feature])

            # height,width=pred_instances.image_size
            # print(pred_instances)
            # print(pred_instances.pred_classes.shape)
            # print(pred_instances.pred_boxes.tensor.shape)
            # print(pred_instances.pred_masks.shape)
            # plt.figure()
            # image = images[0].permute(1, 2, 0).data.cpu().numpy()#[:, :, ::-1]
            # for i in range(pred_instances[0].pred_classes.shape[0]):
            #     # mask=pred_instances.pred_masks[i].data.cpu().numpy()
            #     box=pred_instances[0].pred_boxes.tensor[i].data.cpu().numpy()
            #     plt.imshow(image)
            #     plt.gca().add_patch(plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,linewidth=1,color='red'))
            #     plt.show()
            # =================pred_instances,pred_box_features

            pred_pair_instances = None
            pred_pair_box_features = None
            pred_pair_instance_nums=[]
            if "pair" in self.relation_head_list or "predicate" in self.relation_head_list:
                pred_pair_instances = generate_pair_instances(pred_instances)
                pred_pair_instance_nums = [len(pred_pair_instance) for pred_pair_instance in pred_pair_instances]
                pred_pair_box_features_mix = self.roi_heads.generate_instance_box_features(features, [pred_pair_instance.pred_pair_boxes for pred_pair_instance in pred_pair_instances])
                pred_pair_box_features = pred_pair_box_features_mix.split(pred_pair_instance_nums)

            # print(pred_pair_instances)
            # print(pred_pair_instances.pred_pair_sub_classes.shape)
            # print(pred_pair_instances.pred_pair_obj_classes.shape)
            # print(pred_pair_instances.pred_pair_boxes.tensor.shape)
            # print(pred_pair_instances.pred_pair_masks.shape)
            # print(pred_pair_instances.pred_pair_locations.shape)
            # print(pred_pair_instances.pred_pair_union_locations.shape)
            # plt.figure()
            # for i in range(pred_pair_instances.pred_pair_masks.shape[0]):
            #     mask = pred_pair_instances.pred_pair_masks[i].data.cpu().numpy()
            #     box = pred_pair_instances.pred_pair_boxes.tensor[i].data.cpu().numpy() * [128. / width, 128. / height,
            #                                                                     128. / width, 128. / height]
            #     plt.imshow(mask)
            #     plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, linewidth=0.3,color='red'))
            #     plt.show()

            #=================thing_instances,(gt_thing_instances),thing_box_features
            #=================stuff_instances,(gt_stuff_instances),stuff_box_features
            #=================features, pred_instance_box_features, pred_pair_box_features
            #=================pred_instance_box_features

            pred_mannual_triplets = None
            if training:
                pred_instances, pred_gt_triplets = map_gt_and_relations(pred_instances, gt_instances, gt_triplets)
                pred_instances = generate_instances_interest(pred_instances, pred_gt_triplets,self.relation_num)
                if "pair" in self.relation_head_list or "predicate" in self.relation_head_list:
                    pred_pair_instances = generate_pairs_interest(pred_instances,pred_pair_instances, pred_gt_triplets, self.relation_num)
                if "predicate" in self.relation_head_list:
                    pred_mannual_triplets = None#generate_mannual_relation(pred_instances, pred_gt_triplets)
            # print("gt_instances")
            # print(gt_instances[0].pred_classes)
            # print(gt_instances[1].pred_classes)
            # print("gt_triplets")
            # print(gt_triplets[0].gt_subject_ids)
            # print(gt_triplets[0].gt_object_ids)
            # print(gt_triplets[0].gt_relation_ids)
            # print(gt_triplets[1].gt_subject_ids)
            # print(gt_triplets[1].gt_object_ids)
            # print(gt_triplets[1].gt_relation_ids)
            # print("pred_instances")
            # print(pred_instances[0].pred_classes)
            # print(pred_instances[0].pred_interest)
            # print(pred_instances[1].pred_classes)
            # print(pred_instances[1].pred_interest)
            # print("pred_gt_triplets")
            # print(pred_gt_triplets[0].gt_subject_ids)
            # print(pred_gt_triplets[0].gt_object_ids)
            # print(pred_gt_triplets[0].gt_relation_ids)
            # print(pred_gt_triplets[1].gt_subject_ids)
            # print(pred_gt_triplets[1].gt_object_ids)
            # print(pred_gt_triplets[1].gt_relation_ids)
            # print("pred_pair_instances")
            # print(pred_pair_instances[0].pred_pair_sub_classes)
            # print(pred_pair_instances[0].pred_pair_obj_classes)
            # print(pred_pair_instances[0].pred_pair_interest)
            # print(torch.argmax(pred_pair_instances[0].pred_pair_gt_predicate_full,dim=1))
            # print(pred_pair_instances[1].pred_pair_sub_classes)
            # print(pred_pair_instances[1].pred_pair_obj_classes)
            # print(pred_pair_instances[1].pred_pair_interest)
            # print(torch.argmax(pred_pair_instances[1].pred_pair_gt_predicate_full, dim=1))
            # exit()

            # if training:
            #     _, _, gt_class_loss, gt_class_metirc = self.instance_encoder(
            #         image_features, gt_instances, gt_box_features, training)
            #     losses['gt_class_loss']=gt_class_loss['pred_class_loss']
            #     metrics['gt_class_tp']=gt_class_metirc['pred_class_tp']
            #     metrics['gt_class_p'] = gt_class_metirc['pred_class_p']
            pred_instance_features, pred_instance_logits, pred_class_loss, pred_class_metirc = self.instance_encoder(
                image_features, pred_instances, pred_box_features, training)
            if pred_instance_logits is not None:
                for i in range(len(pred_instance_nums)):
                    pred_instances[i].pred_class_logits = pred_instance_logits[i]
                    instance_num = len(pred_instances[i])
                    subject_logits = pred_instance_logits[i][:, 1:].repeat(instance_num, 1, 1).permute(1, 0,2).contiguous().view(instance_num * instance_num, -1)
                    object_logits = pred_instance_logits[i][:, 1:].repeat(instance_num, 1, 1).contiguous().view(instance_num * instance_num, -1)
                    pred_pair_instances[i].pred_subject_logits=subject_logits
                    pred_pair_instances[i].pred_object_logits = object_logits

            if training:
                losses.update(pred_class_loss)
                metrics.update(pred_class_metirc)

            pred_pair_instance_features=None
            pred_pair_predicate_features = None
            if "pair" in self.relation_head_list or "predicate" in self.relation_head_list:
                pred_pair_instance_features, _, _, _ = self.instance_encoder(
                    image_features, pred_pair_instances, pred_pair_box_features, training=False)

                pred_pair_predicate_features = []
                for k in range(len(pred_instance_nums)):
                    pred_instance=pred_instances[k]
                    pred_instance_feature=pred_instance_features[k]
                    pred_pair_instance_feature=pred_pair_instance_features[k]

                    pred_pair_predicate_feature = []
                    for i in range(len(pred_instance)):
                        for j in range(len(pred_instance)):
                            pred_pair_predicate_feature.append(torch.cat([pred_instance_feature[i] - pred_instance_feature[j],pred_pair_instance_feature[i * pred_instance_feature.shape[0] + j]]))
                    pred_pair_predicate_feature = torch.stack(pred_pair_predicate_feature)
                    pred_pair_predicate_features.append(pred_pair_predicate_feature)
            # ==================pred_instances,pred_pair_instances,pred_pair_box_features,pred_pair_predicate_features

            # del images
            # del gt_thing_instances
            # del gt_stuff_instances
            # del gt_triplets
            # del batched_inputs[0]['image']
            # del batched_inputs[0]['instance_ids']
            # del batched_inputs[0]['stuff_instance_ids']
            # del batched_inputs[0]['instances']
            # del batched_inputs[0]['stuffs']
            # del batched_inputs[0]['triplets']
            # del batched_inputs[0]['sem_seg']
            # del detector_results
            # del thing_instances
            # del sem_seg_results
            # del stuff_instances

            relation_results, relation_losses, relation_metrics = self.relation_heads(image_features,pred_instances, pred_pair_instances,
                                                                                      pred_instance_features, pred_pair_instance_features, pred_pair_predicate_features,
                                                                                      pred_mannual_triplets,training,iteration)
            losses.update(relation_losses)
            metrics.update(relation_metrics)
            for name in losses:
                losses[name]=losses[name]*1.0/len(batched_inputs)
            return pred_instances, relation_results, losses, metrics

def combine_semantic_and_instance_outputs(
        thing_id_map,
        stuff_id_map,
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
    mode="panoptic"
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_confidence_threshold:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        box=extract_bbox(mask).to(panoptic_seg.device)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id

        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "class_id": thing_id_map[instance_results.pred_classes[inst_id].item()],
                "instance_id": inst_id.item(),
                "mask": mask,
                "box": box
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_limit:
            continue

        box = extract_bbox(mask).to(panoptic_seg.device)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": semantic_label,
                "class_id": stuff_id_map[semantic_label],
                "area": mask_area,
                "mask": mask,
                "box": box
            }
        )

    return panoptic_seg, segments_info

# def combine_semantic_and_instance_outputs(
#     instance_results,
#     overlap_threshold,
#     instances_confidence_threshold,
#     mode="panoptic"
# ):
#     """
#     Implement a simple combining logic following
#     "combine_semantic_and_instance_predictions.py" in panopticapi
#     to produce panoptic segmentation outputs.
#
#     Args:
#         instance_results: output of :func:`detector_postprocess`.
#         semantic_results: an (H, W) tensor, each is the contiguous semantic
#             category id
#
#     Returns:
#         panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
#         segments_info (list[dict]): Describe each segment in `panoptic_seg`.
#             Each dict contains keys "id", "category_id", "isthing".
#     """
#     panoptic_seg = torch.zeros((instance_results.image_height,instance_results.image_width), dtype=torch.int32)
#
#     # sort instance outputs by scores
#     sorted_inds = torch.argsort(-instance_results.scores)
#
#     current_segment_id = 0
#     segments_info = []
#
#     instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)
#
#     # Add instances one-by-one, check for overlaps with existing ones
#     for inst_id in sorted_inds:
#         score = instance_results.scores[inst_id].item()
#         if score < instances_confidence_threshold:
#             break
#         mask = instance_masks[inst_id]  # H,W
#         mask_area = mask.sum().item()
#
#         if mask_area == 0:
#             continue
#
#         intersect = (mask > 0) & (panoptic_seg > 0)
#         intersect_area = intersect.sum().item()
#
#         if intersect_area * 1.0 / mask_area > overlap_threshold:
#             continue
#
#         if intersect_area > 0:
#             mask = mask & (panoptic_seg == 0)
#
#         box=extract_bbox(mask).to(panoptic_seg.device)
#
#         current_segment_id += 1
#         panoptic_seg[mask] = current_segment_id
#
#         segments_info.append(
#             {
#                 "id": current_segment_id,
#                 "isthing": True,
#                 "score": score,
#                 "category_id": instance_results.pred_classes[inst_id].item(),
#                 "class_id": thing_id_map[instance_results.pred_classes[inst_id].item()],
#                 "instance_id": inst_id.item(),
#                 "mask": mask,
#                 "box": box
#             }
#         )





