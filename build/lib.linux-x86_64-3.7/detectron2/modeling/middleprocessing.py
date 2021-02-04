# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F

from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances, Boxes, Triplets
from detectron2.utils.torch_utils import extract_bbox, extract_bboxes, mask_iou, box_iou, boxes_iou

def generate_thing_instances(id_map, thing_instances):
    new_thing_instances=[]
    keep_areas=[]
    for i in range(len(thing_instances)):
        thing_instance=thing_instances[i]
        new_thing_instance=Instances(thing_instance.image_size)
        if len(thing_instance)>0:
            pred_masks = paste_masks_in_image(
                thing_instance.pred_masks[:, 0, :, :],  # N, 1, M, M
                thing_instance.pred_boxes,
                thing_instance.image_size
            )
            # upsample_pred_masks = F.upsample(pred_masks.float().unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
            # areas=upsample_pred_masks.sum(dim=1).sum(dim=1)
            areas = pred_masks.sum(dim=1).sum(dim=1)
            area_keep=torch.where(areas>0)
            keep_areas.append(area_keep)
            pred_keep_masks=pred_masks[area_keep]
            new_thing_instance.pred_classes=thing_instance.pred_classes[area_keep]
            new_thing_instance.pred_boxes=Boxes(extract_bboxes(pred_keep_masks)).to(new_thing_instance.pred_classes.device)
            # new_thing_instance.pred_masks=upsample_pred_masks[area_keep]
            # new_thing_instance.pred_masks = pred_keep_masks

            pred_classes=[]
            for pred_class in new_thing_instance.pred_classes:
                pred_classes.append(id_map[pred_class.item()])
            pred_classes=torch.IntTensor(pred_classes).to(new_thing_instance.pred_classes.device)
            new_thing_instance.pred_classes=pred_classes
        else:
            keep_areas.append([])
        new_thing_instances.append(new_thing_instance)
    del thing_instances

    return new_thing_instances, keep_areas

def generate_stuff_instances(id_map, sem_seg_results,correct_sizes):
    stuff_instances = []
    for i in range(len(correct_sizes)):
        correct_size=correct_sizes[i]
        sem_seg_result=sem_seg_results[i, : ,:correct_size[0], : correct_size[1]]

        stuff_instance = Instances(correct_size)
        sem_seg=sem_seg_result.argmax(dim=0)
        pred_classes_before=torch.unique(sem_seg)[1:]
        if pred_classes_before.shape[0] > 0:
            pred_masks=[]
            pred_boxes=[]
            pred_classes=[]
            for pred_class in pred_classes_before:
                mask=sem_seg==pred_class
                # upsample_pred_masks = F.upsample(mask.float().unsqueeze(0).unsqueeze(0), size=(128, 128),mode='bilinear').squeeze(0).squeeze(0)
                # if torch.sum(torch.sum(upsample_pred_masks>0,dim=1))<=0:
                #     continue
                # pred_masks.append(upsample_pred_masks)
                if torch.sum(torch.sum(mask,dim=1))<=1000:
                    continue
                # pred_masks.append(mask)
                box = extract_bbox(mask)
                pred_boxes.append(box)
                pred_class=id_map[pred_class.item()]
                pred_classes.append(pred_class)
            if len(pred_boxes)>0:
                pred_boxes=torch.stack(pred_boxes).to(pred_classes_before.device)
                # pred_masks=torch.stack(pred_masks)
                pred_classes=torch.IntTensor(pred_classes).to(pred_classes_before.device)

                # stuff_instance.pred_masks = pred_masks
                stuff_instance.pred_boxes = Boxes(pred_boxes)
                stuff_instance.pred_classes = pred_classes
            else:
                stuff_instance.pred_boxes = Boxes(torch.IntTensor().to(sem_seg.device))
                stuff_instance.pred_classes = torch.IntTensor().to(sem_seg.device)
                # stuff_instance.pred_masks = torch.FloatTensor().to(sem_seg.device)
        else:
            # stuff_instance.pred_masks = torch.FloatTensor().to(sem_seg.device)
            stuff_instance.pred_boxes = Boxes(torch.IntTensor().to(sem_seg.device))
            stuff_instance.pred_classes = torch.IntTensor().to(sem_seg.device)
        stuff_instances.append(stuff_instance)
    return stuff_instances

def generate_parts(id_map, parts):
    new_parts=[]
    keep_areas=[]
    for i in range(len(parts)):
        part=parts[i]
        new_part=Part(part.image_size)
        if len(part)>0:
            pred_masks = paste_masks_in_image(
                part.pred_masks[:, 0, :, :],  # N, 1, M, M
                part.pred_boxes,
                part.image_size
            )
            # upsample_pred_masks = F.upsample(pred_masks.float().unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
            # areas=upsample_pred_masks.sum(dim=1).sum(dim=1)
            areas = pred_masks.sum(dim=1).sum(dim=1)
            area_keep=torch.where(areas>0)
            keep_areas.append(area_keep)
            pred_keep_masks=pred_masks[area_keep]
            new_part.pred_classes=part.pred_classes[area_keep]
            new_part.pred_boxes=Boxes(extract_bboxes(pred_keep_masks)).to(new_part.pred_classes.device)
            # new_thing_instance.pred_masks=upsample_pred_masks[area_keep]
            # new_thing_instance.pred_masks = pred_keep_masks

            pred_classes=[]
            for pred_class in new_part.pred_classes:
                pred_classes.append(id_map[pred_class.item()])
            pred_classes=torch.IntTensor(pred_classes).to(new_part.pred_classes.device)
            new_part.pred_classes=pred_classes
        else:
            keep_areas.append([])
        new_parts.append(new_part)
    del parts

    return new_parts, keep_areas

def generate_gt_instances(gt_thing_instances,gt_stuff_instances):
    for gt_thing_instance in gt_thing_instances:
        if gt_thing_instance is not None:
            gt_thing_instance.pred_classes = gt_thing_instance.gt_class_ids
            gt_thing_instance.pred_boxes = gt_thing_instance.gt_boxes
            # if len(gt_thing_instances)>0:
            #     gt_thing_instances.pred_masks = F.upsample(gt_thing_instances.gt_masks.tensor.float().unsqueeze(1),size=(128, 128), mode='bilinear').squeeze(1)

    for gt_stuff_instance in gt_stuff_instances:
        if gt_stuff_instance is not None:
            gt_stuff_instance.pred_classes = gt_stuff_instance.gt_class_ids
            gt_stuff_instance.pred_boxes = gt_stuff_instance.gt_boxes
            # if len(gt_stuff_instances)>0:
            #     gt_stuff_instances.pred_masks = F.upsample(gt_stuff_instances.gt_masks.tensor.float().unsqueeze(1), size=(128, 128),mode='bilinear').squeeze(1)
    gt_instances=generate_instances(gt_thing_instances, gt_stuff_instances)
    for gt_instance in gt_instances:
        gt_instance.pred_gt_classes=gt_instance.pred_classes
    del gt_thing_instances
    del gt_stuff_instances
    return gt_instances

def generate_instances(pred_thing_instances, pred_stuff_instances):
    pred_instances=[]
    for i in range(len(pred_thing_instances)):
        pred_thing_instance = pred_thing_instances[i]
        pred_stuff_instance = pred_stuff_instances[i]

        pred_instance = Instances(pred_thing_instance.image_size)
        if len(pred_thing_instance) > 0 and len(pred_stuff_instance) > 0:
            pred_classes = torch.cat([pred_thing_instance.pred_classes, pred_stuff_instance.pred_classes])
            pred_boxes = torch.cat([pred_thing_instance.pred_boxes.tensor, pred_stuff_instance.pred_boxes.tensor])
            # pred_masks = torch.cat([pred_thing_instance.pred_masks, pred_stuff_instance.pred_masks])
        elif len(pred_thing_instance) > 0:
            pred_classes = pred_thing_instance.pred_classes
            pred_boxes = pred_thing_instance.pred_boxes.tensor
            # pred_masks = pred_thing_instance.pred_masks
        elif len(pred_stuff_instance) > 0:
            pred_classes = pred_stuff_instance.pred_classes
            pred_boxes = pred_stuff_instance.pred_boxes.tensor
            # pred_masks = pred_stuff_instance.pred_masks
        else:
            pred_instances.append(pred_instance)
            continue
        pred_instance.pred_classes = pred_classes
        pred_instance.pred_boxes = Boxes(pred_boxes)
        # pred_instance.pred_masks = pred_masks

        image_height, image_width = pred_instance.image_size
        instance_locations = torch.stack([pred_boxes[:, 0] / image_width,
                                          pred_boxes[:, 1] / image_height,
                                          (pred_boxes[:, 2] - image_width) / image_width,
                                          (pred_boxes[:, 3] - image_height) / image_height], dim=1)
        pred_instance.pred_locations = instance_locations
        pred_instances.append(pred_instance)
    del pred_thing_instances
    del pred_stuff_instances
    return pred_instances

def generate_instances_interest(pred_instances, gt_triplets,relation_num):
    for i in range(len(pred_instances)):
        pred_instance=pred_instances[i]
        gt_triplet=gt_triplets[i]

        pred_interest=torch.zeros_like(pred_instance.pred_classes)
        interest_instance_ids=torch.cat([gt_triplet.gt_subject_ids,gt_triplet.gt_object_ids])
        pred_interest[interest_instance_ids]=1
        pred_instance.pred_interest=pred_interest

        subpred_interest = torch.zeros((len(pred_instance)*relation_num)).to(pred_interest.device)
        subpred_interest[gt_triplet.gt_subject_ids * relation_num + gt_triplet.gt_relation_ids]=1
        predobj_interest = torch.zeros((len(pred_instance)*relation_num)).to(pred_interest.device)
        predobj_interest[gt_triplet.gt_object_ids * relation_num + gt_triplet.gt_relation_ids] = 1
        pred_instance.pred_subpred_interest = subpred_interest.view(len(pred_instance),relation_num)
        pred_instance.pred_predobj_interest = predobj_interest.view(len(pred_instance),relation_num)
    return pred_instances

def generate_pair_instances(pred_instances):
    pred_pair_instances=[]
    for pred_instance in pred_instances:
        pred_pair_instance = Instances(pred_instance.image_size)
        instance_num=len(pred_instance)
        pred_classes = pred_instance.pred_classes
        pred_boxes = pred_instance.pred_boxes.tensor
        # pred_masks=pred_instance.pred_masks

        image_height, image_width = pred_instance.image_size
        pred_pair_sub_classes=pred_classes.repeat(instance_num,1).permute(1,0).flatten()
        pred_pair_obj_classes=pred_classes.repeat(instance_num)

        pred_pair_sub_boxes=pred_boxes.repeat(instance_num,1,1).permute(1,0,2).contiguous().view(-1,4)
        pred_pair_obj_boxes=pred_boxes.repeat(instance_num,1)
        sub_boxes_x1=pred_pair_sub_boxes[:, 0]
        obj_boxes_x1=pred_pair_obj_boxes[:, 0]
        sub_boxes_y1=pred_pair_sub_boxes[:, 1]
        obj_boxes_y1=pred_pair_obj_boxes[:, 1]
        sub_boxes_x2=pred_pair_sub_boxes[:, 2]
        obj_boxes_x2=pred_pair_obj_boxes[:, 2]
        sub_boxes_y2=pred_pair_sub_boxes[:, 3]
        obj_boxes_y2=pred_pair_obj_boxes[:, 3]
        pair_boxes_x1=torch.min(sub_boxes_x1,obj_boxes_x1)
        pair_boxes_y1=torch.min(sub_boxes_y1,obj_boxes_y1)
        pair_boxes_x2=torch.max(sub_boxes_x2,obj_boxes_x2)
        pair_boxes_y2=torch.max(sub_boxes_y2,obj_boxes_y2)
        pred_pair_boxes=torch.stack([pair_boxes_x1,pair_boxes_y1,pair_boxes_x2,pair_boxes_y2],dim=1)
        pred_pair_locations=torch.stack([(sub_boxes_x1 - 0) / image_width,
                                            (sub_boxes_y1 - 0) / image_height,
                                            (sub_boxes_x2 - image_width) / image_width,
                                            (sub_boxes_y2 - image_height) / image_height,
                                            (obj_boxes_x1 - 0) / image_width,
                                            (obj_boxes_y1 - 0) / image_height,
                                            (obj_boxes_x2 - image_width) / image_width,
                                            (obj_boxes_y2 - image_height) / image_height],dim=1)
        pair_width = pair_boxes_x2 - pair_boxes_x1
        pair_height = pair_boxes_y2 - pair_boxes_y1
        pred_pair_union_locations=torch.stack([(sub_boxes_x1 - pair_boxes_x1) / pair_width,
                                                  (sub_boxes_y1 - pair_boxes_y1) / pair_height,
                                                  (sub_boxes_x2 - pair_boxes_x2) / pair_width,
                                                  (sub_boxes_y2 - pair_boxes_y2) / pair_height,
                                                  (obj_boxes_x1 - pair_boxes_x1) / pair_width,
                                                  (obj_boxes_y1 - pair_boxes_y1) / pair_height,
                                                  (obj_boxes_x2 - pair_boxes_x2) / pair_width,
                                                  (obj_boxes_y2 - pair_boxes_y2) / pair_height],dim=1)
        pred_pair_iou=boxes_iou(pred_pair_sub_boxes, pred_pair_obj_boxes)
        pred_pair_left_boxes=pred_pair_boxes.repeat(instance_num*instance_num,1,1).permute(1,0,2).contiguous().view(-1,4)
        pred_pair_right_boxes=pred_pair_boxes.repeat(instance_num*instance_num,1)
        pred_union_iou=boxes_iou(pred_pair_left_boxes, pred_pair_right_boxes).view(instance_num*instance_num,instance_num*instance_num)

        left=torch.arange(0, instance_num).repeat(instance_num,1).permute(1,0).flatten().to(pred_classes.device)
        right=torch.arange(0, instance_num).repeat(instance_num).flatten().to(pred_classes.device)
        lr_loc=torch.stack([left,right],dim=1)
        pred_pair_instance_relate_matrix=torch.zeros(instance_num*instance_num,instance_num).to(pred_classes.device).scatter_(1,lr_loc,1.0)


        # pred_pair_sub_classes = []
        # pred_pair_obj_classes = []
        # pred_pair_boxes = []
        # # pred_pair_masks=[]
        # pred_pair_locations = []
        # pred_pair_union_locations = []
        # pred_pair_iou = []
        # pred_subobj_ids = []
        # pred_pair_instance_relate_matrix = []
        # for i in range(pred_classes.shape[0]):
        #     sub_class = pred_classes[i].item()
        #     sub_box = pred_boxes[i]
        #     sub_x1 = sub_box[0].item()
        #     sub_y1 = sub_box[1].item()
        #     sub_x2 = sub_box[2].item()
        #     sub_y2 = sub_box[3].item()
        #     # sub_mask = pred_masks[i]
        #     for j in range(pred_classes.shape[0]):
        #         obj_class = pred_classes[j].item()
        #         pred_pair_sub_classes.append(sub_class)
        #         pred_pair_obj_classes.append(obj_class)
        #
        #         obj_box = pred_boxes[j]
        #         obj_x1 = obj_box[0].item()
        #         obj_y1 = obj_box[1].item()
        #         obj_x2 = obj_box[2].item()
        #         obj_y2 = obj_box[3].item()
        #
        #         pair_x1 = min(sub_x1, obj_x1)
        #         pair_y1 = min(sub_y1, obj_y1)
        #         pair_x2 = max(sub_x2, obj_x2)
        #         pair_y2 = max(sub_y2, obj_y2)
        #
        #         pair_width = pair_x2 - pair_x1
        #         pair_height = pair_y2 - pair_y1
        #
        #         pred_pair_boxes.append([pair_x1, pair_y1, pair_x2, pair_y2])
        #         pred_pair_locations.append([(sub_x1 - 0) / image_width,
        #                                     (sub_y1 - 0) / image_height,
        #                                     (sub_x2 - image_width) / image_width,
        #                                     (sub_y2 - image_height) / image_height,
        #                                     (obj_x1 - 0) / image_width,
        #                                     (obj_y1 - 0) / image_height,
        #                                     (obj_x2 - image_width) / image_width,
        #                                     (obj_y2 - image_height) / image_height])
        #         pred_pair_union_locations.append([(sub_x1 - pair_x1) / pair_width,
        #                                           (sub_y1 - pair_y1) / pair_height,
        #                                           (sub_x2 - pair_x2) / pair_width,
        #                                           (sub_y2 - pair_y2) / pair_height,
        #                                           (obj_x1 - pair_x1) / pair_width,
        #                                           (obj_y1 - pair_y1) / pair_height,
        #                                           (obj_x2 - pair_x2) / pair_width,
        #                                           (obj_y2 - pair_y2) / pair_height])
        #         pred_pair_iou.append(box_iou(sub_box, obj_box))
        #         pred_subobj_ids.append([i, j])
        #         instance_vector = torch.zeros(pred_classes.shape[0]).to(pred_classes.device)
        #         instance_vector[i] = 1
        #         instance_vector[j] = 1
        #         pred_pair_instance_relate_matrix.append(instance_vector)
        #         # obj_mask=pred_masks[j]
        #         # pred_pair_masks.append(sub_mask+obj_mask)
        # pred_pair_instance.pred_pair_sub_classes = torch.IntTensor(pred_pair_sub_classes).to(pred_classes.device)
        # pred_pair_instance.pred_pair_obj_classes = torch.IntTensor(pred_pair_obj_classes).to(pred_classes.device)
        # pred_pair_instance.pred_pair_boxes = Boxes(torch.FloatTensor(pred_pair_boxes)).to(pred_classes.device)
        # # pred_pair_instance.pred_pair_masks=torch.stack(pred_pair_masks)
        # pred_pair_instance.pred_pair_locations = torch.FloatTensor(pred_pair_locations).to(pred_classes.device)
        # pred_pair_instance.pred_pair_union_locations = torch.FloatTensor(pred_pair_union_locations).to(pred_classes.device)
        # pred_pair_instance.pred_pair_iou = torch.FloatTensor(pred_pair_iou).to(pred_classes.device)
        # pred_subobj_ids = torch.IntTensor(pred_subobj_ids).to(pred_classes.device)
        # pred_pair_relate_matrix = []
        # for i in range(pred_subobj_ids.shape[0]):
        #     pred_pair_relate_matrix.append((pred_subobj_ids[i] == pred_subobj_ids).sum(1))
        # pred_union_iou=[]
        # for box1 in pred_pair_instance.pred_pair_boxes.tensor:
        #     pred_union_iou_row=[]
        #     for box2 in pred_pair_instance.pred_pair_boxes.tensor:
        #         pred_union_iou_row.append(box_iou(box1, box2))
        #     pred_union_iou.append(pred_union_iou_row)
        # pred_pair_instance.pred_union_iou=torch.FloatTensor(pred_union_iou).to(pred_classes.device)
        # pred_pair_relate_matrix = torch.stack(pred_pair_relate_matrix)
        # pred_pair_instance.pred_pair_relate_matrix = pred_pair_relate_matrix.float()
        # pred_pair_instance_relate_matrix = torch.stack(pred_pair_instance_relate_matrix)
        # pred_pair_instance.pred_pair_instance_relate_matrix = pred_pair_instance_relate_matrix.float()
        pred_pair_instance.pred_pair_sub_classes=pred_pair_sub_classes
        pred_pair_instance.pred_pair_obj_classes=pred_pair_obj_classes
        pred_pair_instance.pred_pair_boxes=Boxes(pred_pair_boxes)
        pred_pair_instance.pred_pair_locations=pred_pair_locations
        pred_pair_instance.pred_pair_union_locations=pred_pair_union_locations
        pred_pair_instance.pred_pair_iou=pred_pair_iou
        pred_pair_instance.pred_union_iou=pred_union_iou
        pred_pair_instance.pred_pair_instance_relate_matrix=pred_pair_instance_relate_matrix

        pred_pair_instances.append(pred_pair_instance)
    return pred_pair_instances

def generate_pairs_interest(gt_instances,pred_pair_instances,gt_triplets,relation_num):
    for i in range(len(gt_instances)):
        gt_instance=gt_instances[i]
        pred_pair_instance=pred_pair_instances[i]
        gt_triplet=gt_triplets[i]
        instance_num=len(gt_instance)

        gt_subject_ids = gt_triplet.gt_subject_ids
        gt_object_ids = gt_triplet.gt_object_ids
        gt_relation_ids = gt_triplet.gt_relation_ids

        pred_pair_interest=torch.zeros_like(pred_pair_instance.pred_pair_sub_classes)
        pred_pair_interest[(gt_subject_ids * instance_num + gt_object_ids).long()]=1
        pred_pair_instance.pred_pair_interest=pred_pair_interest

        pred_pair_gt_predicate_full=torch.zeros(len(pred_pair_instance),relation_num).to(pred_pair_instance.pred_pair_sub_classes.device)
        pred_pair_gt_predicate_full=pred_pair_gt_predicate_full.flatten()
        pred_pair_gt_predicate_full[(gt_subject_ids * instance_num*relation_num + gt_object_ids*relation_num+gt_relation_ids).long()]=1
        pred_pair_instance.pred_pair_gt_predicate_full = pred_pair_gt_predicate_full.view(len(pred_pair_instance),relation_num)

    return pred_pair_instances

def generate_mannual_relation(gt_instances, gt_triplets):
    mannual_triplets=[]
    for i in range(len(gt_instances)):
        gt_instance=gt_instances[i]
        gt_triplet = gt_triplets[i]

        gt_classes=gt_instance.pred_classes
        gt_boxes=gt_instance.pred_boxes.tensor
        # gt_masks=gt_instance.pred_masks

        # =============triplet
        mannual_triplet = Triplets(gt_instance.image_size)
        sub_ids = gt_triplet.gt_subject_ids
        obj_ids = gt_triplet.gt_object_ids
        relation_ids = gt_triplet.gt_relation_ids

        eff_ids = []
        for sub_id in sub_ids:
            if sub_id.item() not in eff_ids:
                eff_ids.append(sub_id.item())
        for obj_id in obj_ids:
            if obj_id.item() not in eff_ids:
                eff_ids.append(obj_id.item())
        eff_ids = torch.IntTensor(eff_ids).to(gt_classes.device)
        mannual_subject_ids = eff_ids.unsqueeze(1).repeat(1, eff_ids.shape[0]).flatten()
        mannual_object_ids = eff_ids.unsqueeze(0).repeat(eff_ids.shape[0], 1).flatten()
        mannual_relation_ids = torch.zeros_like(mannual_subject_ids)
        eff_ids_dict = {}
        for i, eff_id in enumerate(eff_ids):
            eff_ids_dict[eff_id.item()] = i
        for i in range(sub_ids.shape[0]):
            sub_id = sub_ids[i].item()
            obj_id = obj_ids[i].item()
            relation_id = relation_ids[i].item()
            mannual_location = eff_ids_dict[sub_id] * len(eff_ids_dict) + eff_ids_dict[obj_id]
            mannual_relation_ids[mannual_location] = relation_id
        mannual_triplet.gt_subject_ids = mannual_subject_ids
        mannual_triplet.gt_object_ids = mannual_object_ids
        mannual_triplet.pred_gt_relation_ids = mannual_relation_ids
        mannual_triplets.append(mannual_triplet)
    return mannual_triplets

# def map_pred_gt_triplets(pred_instances, gt_instances):
#     pred_gt, gt_pred_dict = map_instances(pred_instances, gt_instances)
    # =============max gt for pred, satisfied preds for gt

    # pred_gt_classes = []
    # for i in range(pred_classes.shape[0]):
    #     gt_id = list(pred_gt[i].keys())[0]
    #     pred_gt_classes.append(gt_classes[gt_id].item())
    # pred_gt_classes = torch.LongTensor(pred_gt_classes).to(pred_classes.device)



# def generate_relation_target(pred_instances,gt_instances, gt_triplets, relation_num):
#
#     pred_gt_classes, pred_gt_interest_triplets, \
#     pred_gt_pair_predicate, \
#     pred_gt_instances_full, \
#     pred_gt_subpreds_full, pred_gt_objpreds_full, pred_gt_predicates_full, \
#     pred_gt_pairs_full, \
#     pred_gt_triplets_full = map_relations(pred_instances,gt_instances,gt_triplets, relation_num, self.relation_head_list)
#     gt_instances, gt_pair_instances, mannual_triplet, gt_pair_predicate, gt_triplets_full = self.generate_mannual_relation(
#         pred_instances, gt_classes, gt_boxes, gt_masks, gt_triplets)
#
#     # =============pred_gt_classes : real class for pred instances [n]
#     # =============pred_gt_pair_predicate : predicates of all pairs [n^2]
#
#     # =============pred_gt_instances_full : interest of all pred instances [n]
#     # =============pred_gt_pairs_full : interest of all pred pairs [n^2]
#     # =============pred_gt_triplets_full : interest of all pred triplets [n^2, 250]
#
#     return pred_gt_classes, pred_gt_interest_triplets, \
#            pred_gt_pair_predicate, gt_pair_predicate, \
#            pred_gt_instances_full, \
#            pred_gt_subpreds_full, pred_gt_objpreds_full, pred_gt_predicates_full, \
#            pred_gt_pairs_full, \
#            pred_gt_triplets_full, gt_triplets_full, \
#            gt_instances, gt_pair_instances, mannual_triplet
#
def map_instances(pred_instances, gt_instances):
    pred_gts=[]
    gt_pred_dicts=[]
    for i in range(len(pred_instances)):
        pred_instance=pred_instances[i]
        gt_instance=gt_instances[i]

        pred_classes = pred_instance.pred_classes
        pred_boxes = pred_instance.pred_boxes.tensor
        gt_classes = gt_instance.pred_classes
        gt_boxes = gt_instance.pred_boxes.tensor
        pred_gt = {}
        gt_pred = {}
        for i in range(pred_boxes.shape[0]):
            pred_class = pred_classes[i].item()
            pred_box = pred_boxes[i]
            pred_gt[i] = []
            for j in range(gt_boxes.shape[0]):
                gt_class = gt_classes[j].item()
                if i == 0:
                    gt_pred[j] = []
                gt_box = gt_boxes[j]
                iou = box_iou(pred_box, gt_box)
                pred_gt[i].append(iou)
                if pred_class == gt_class:
                    gt_pred[j].append(iou)
                else:
                    gt_pred[j].append(-1)

        for pred_id in pred_gt:
            pred_gt[pred_id] = {np.argmax(np.array(pred_gt[pred_id])): np.max(np.array(pred_gt[pred_id]))}

        gt_pred_dict = {}
        for gt_id in gt_pred:
            gt_pred_dict[gt_id] = {}
            for pred_id, iou in enumerate(gt_pred[gt_id]):
                if iou > 0.5:
                    gt_pred_dict[gt_id][pred_id] = iou

        pred_gts.append(pred_gt)
        gt_pred_dicts.append(gt_pred_dict)
    return pred_gts, gt_pred_dicts

def map_gt_and_relations(pred_instances, gt_instances, gt_triplets):
    pred_gts, gt_pred_dicts = map_instances(pred_instances, gt_instances)
    # =============max gt for pred, satisfied preds for gt
    pred_gt_interest_triplets=[]
    for i in range(len(pred_instances)):
        pred_instance = pred_instances[i]
        gt_instance = gt_instances[i]
        pred_gt=pred_gts[i]
        gt_pred_dict=gt_pred_dicts[i]
        gt_triplet=gt_triplets[i]

        pred_classes = pred_instance.pred_classes
        gt_classes = gt_instance.pred_classes
        pred_gt_classes = []
        for i in range(pred_classes.shape[0]):
            gt_id = list(pred_gt[i].keys())[0]
            pred_gt_classes.append(gt_classes[gt_id].item())
        pred_gt_classes = torch.LongTensor(pred_gt_classes).to(pred_classes.device)
        pred_instance.pred_gt_classes = pred_gt_classes

        # =============real class for pred instances

        gt_subject_ids = gt_triplet.gt_subject_ids
        gt_object_ids = gt_triplet.gt_object_ids
        gt_relation_ids = gt_triplet.gt_relation_ids

        pred_gt_subject_ids = []
        pred_gt_object_ids = []
        pred_gt_relation_ids = []
        pred_gt_interest_ids = []  # interest instance
        for i in range(len(gt_subject_ids)):
            gt_subject_id = gt_subject_ids[i].item()
            gt_object_id = gt_object_ids[i].item()
            gt_relation_id = gt_relation_ids[i].item()
            pred_subjects = gt_pred_dict[gt_subject_id]
            pred_objects = gt_pred_dict[gt_object_id]
            # satisfied triplets for one gt triplet
            for pred_subject_id in pred_subjects:
                for pred_object_id in pred_objects:
                    pred_gt_subject_ids.append(pred_subject_id)
                    pred_gt_object_ids.append(pred_object_id)
                    pred_gt_relation_ids.append(gt_relation_id)
                    if pred_subject_id not in pred_gt_interest_ids:
                        pred_gt_interest_ids.append(pred_subject_id)
                    if pred_object_id not in pred_gt_interest_ids:
                        pred_gt_interest_ids.append(pred_object_id)

        pred_gt_interest_triplet = Triplets(pred_instance.image_size)
        pred_gt_interest_triplet.gt_subject_ids = torch.LongTensor(pred_gt_subject_ids)
        pred_gt_interest_triplet.gt_object_ids = torch.LongTensor(pred_gt_object_ids)
        pred_gt_interest_triplet.gt_relation_ids = torch.LongTensor(pred_gt_relation_ids)
        pred_gt_interest_triplets.append(pred_gt_interest_triplet)

    return pred_instances, pred_gt_interest_triplets

# def make_relation_target(instances, triplets):
#     pred_gt_instances_full = None
#     if "instance" in mode:
#         pred_gt_instances_full = []
#         for i in range(pred_classes.shape[0]):
#             if i in pred_gt_interest_ids:
#                 pred_gt_instances_full.append(1)
#             else:
#                 pred_gt_instances_full.append(0)
#         pred_gt_instances_full = torch.FloatTensor(pred_gt_instances_full).to(pred_classes.device)
#
#     # =============interest of all pred instances [n]
#
#     pred_gt_pairs_full = None
#     if "pair" in mode:
#         pred_gt_pairs_full = torch.zeros((pred_classes.shape[0] * pred_classes.shape[0])).to(pred_classes.device)
#         for i in range(pred_gt_interest_triplets.pred_subject_ids.shape[0]):
#             pred_gt_sub_id = pred_gt_interest_triplets.pred_subject_ids[i]
#             pred_gt_obj_id = pred_gt_interest_triplets.pred_object_ids[i]
#             pred_gt_relation_id = pred_gt_interest_triplets.pred_gt_relation_ids[i]
#             pred_gt_pairs_full[pred_gt_sub_id * pred_classes.shape[0] + pred_gt_obj_id] = 1
#
#     # =============interest of all pred pairs [n^2]
#
#     pred_gt_triplets_full = None
#     pred_gt_pair_predicate = None
#     if "predicate" in mode:
#         pred_gt_triplets_full = torch.zeros((pred_classes.shape[0] * pred_classes.shape[0], relation_num)).to(
#             pred_classes.device)
#         pred_gt_pair_predicate = torch.zeros((pred_classes.shape[0] * pred_classes.shape[0])).to(
#             pred_classes.device).long()
#         # for i in range(pred_gt_triplets_full.shape[0]):
#         #     pred_gt_triplets_full[i][0] = 1
#         for i in range(pred_gt_interest_triplets.pred_subject_ids.shape[0]):
#             pred_gt_sub_id = pred_gt_interest_triplets.pred_subject_ids[i]
#             pred_gt_obj_id = pred_gt_interest_triplets.pred_object_ids[i]
#             pred_gt_relation_id = pred_gt_interest_triplets.pred_gt_relation_ids[i]
#             # print(pred_sub_id*pred_classes.shape[0]+pred_obj_id)
#             pred_gt_triplets_full[pred_gt_sub_id * pred_classes.shape[0] + pred_gt_obj_id][pred_gt_relation_id] = 1
#             pred_gt_triplets_full[pred_gt_sub_id * pred_classes.shape[0] + pred_gt_obj_id][0] = 0
#             pred_gt_pair_predicate[pred_gt_sub_id * pred_classes.shape[0] + pred_gt_obj_id] = pred_gt_relation_id
#
#     # =============interest of all pred triplets [n^2, 250] & predicates of all pairs [n^2]
#
#     pred_gt_subpred_full = None
#     pred_gt_predobj_full = None
#     pred_gt_predicates_full = None
#     if "subpredobj" in mode:
#         pred_gt_subpred_full = torch.zeros(pred_classes.shape[0], relation_num).to(pred_classes.device)
#         pred_gt_predobj_full = torch.zeros(pred_classes.shape[0], relation_num).to(pred_classes.device)
#         # pred_gt_predicates_full = torch.zeros(relation_num).to(pred_classes.device)
#         for i in range(pred_classes.shape[0]):
#             for j in range(pred_classes.shape[0]):
#                 pred_gt_sub_id = pred_gt_interest_triplets.pred_subject_ids[i]
#             pred_gt_obj_id = pred_gt_interest_triplets.pred_object_ids[i]
#             pred_gt_relation_id = pred_gt_interest_triplets.pred_gt_relation_ids[i]
#             pred_gt_subpred_full[pred_gt_sub_id * pred_classes.shape[0] + pred_gt_relation_id] = 1
#             pred_gt_predobj_full[pred_gt_obj_id * pred_classes.shape[0] + pred_gt_relation_id] = 1
#             # pred_gt_predicates_full[pred_gt_relation_id] = 1
#
#     # =============interest of all subpred, objpred, predicates