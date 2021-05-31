# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import time
import datetime
from collections import OrderedDict
import json
import numpy as np
from PIL import Image

from pycocotools import mask as maskUtils
from detectron2.data.detection_utils import read_image
from panopticapi.utils import IdGenerator
from detectron2.structures import ImageList, Instances, Boxes, BitMasks
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from torch.nn.parallel import DistributedDataParallel

import pickle
from fvcore.common.file_io import PathManager

from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.np_utils import extract_bbox
import detectron2.utils.comm as comm
from detectron2.engine.defaults import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

logger = logging.getLogger("detectron2")

def do_panoptic_test(cfg, model):
    categoryies=json.load(open("../data/panoptic_coco_categories.json",'r'))
    categoryies_dict={}
    for category in categoryies:
        categoryies_dict[category['id']]=category
    id_generator=IdGenerator(categoryies_dict)
    image_dict = {}
    error_list=[]
    for dataset_name in ['viroi_test']:#,'viroi_train']:#cfg.DATASETS.TRAIN:#cfg.DATASETS.TEST:
        # data_loader = build_detection_test_loader(cfg, dataset_name)
        thing_id_map = MetadataCatalog.get(dataset_name).get("thing_contiguous_id_to_class_id")
        stuff_id_map = MetadataCatalog.get(dataset_name).get("stuff_contiguous_id_to_class_id")
        test_images_dict=json.load(open(MetadataCatalog.get(dataset_name).get("instance_json_file"),'r'))
        image_path = MetadataCatalog.get(dataset_name).get("image_path")

        predictor=DefaultPredictor(cfg)

        total = len(test_images_dict)
        count=0
        # for idx, inputs in enumerate(data_loader):
        for image_id in test_images_dict:
            image_info=test_images_dict[image_id]
            img=read_image(image_path+"/"+image_info['image_name'],format="BGR")
            count+=1
            print(str(count)+"/"+str(total))
            if True:
            # try:
                # print(inputs[0])
                # predictions = model(inputs, "panoptic")[0]  # 'sem_seg', 'instances', 'panoptic_seg'
                predictions = predictor(img,0)
                panoptic_seg, segments_info = predictions["panoptic_seg"]  # seg, info
                panoptic_seg=panoptic_seg.data.cpu().numpy()

                panoptic_color_seg = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1], 3)) #tensor
                instance_dict={}
                for info in segments_info:
                    if 'score' in info:
                        del info['score']
                    if 'area' in info:
                        del info['area']
                    bbox = info['box']  # x1,y1,x2,y2->y1,x1,y2,x2
                    info['bbox'] = [int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]
                    del info['box']

                    class_id=info['class_id']
                    del info['category_id']

                    mask = info['mask'].data.cpu().numpy()
                    mask = np.asfortranarray(mask)
                    segmentation = maskUtils.encode(mask)
                    segmentation['counts'] = segmentation['counts'].decode('utf8')
                    info['segmentation'] = segmentation
                    instance_id, panoptic_color_seg[mask] = id_generator.get_id_and_color(categoryies[class_id - 1]['id'])
                    info['instance_id'] = instance_id
                    del info['mask']

                    instance_dict[str(instance_id)]=info
                image_dict[image_id]={'instances':instance_dict,
                                    "image_id":image_info['image_id'],
                                    "height":image_info['height'],
                                    "width":image_info['width'],
                                    "image_name":image_info['image_name']
                                      }
                # print(image_dict)
                Image.fromarray(panoptic_color_seg.astype(np.uint8)).save("../data/detectron2_panoptic/"+image_info["image_name"].replace("jpg","png"))
            # except:
            #     print("ERROR - "+image_info['image_name'])
            #     error_list.append(image_info['image_name'])
        json.dump(image_dict,open("../data/viroi_json/detectron2_"+dataset_name+"_images_dict.json",'w'))
    json.dump(error_list,open("error_list.json",'w'))

def do_relation_test(cfg, model):
    model.train()
    for param in model.named_parameters():
        param[1].requires_grad=False

    data_loader = build_detection_test_loader(cfg,cfg.DATASETS.TEST[0])
    start_iter=0
    max_iter=len(data_loader)
    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    metrics_sum_dict = {
        'relation_cls_tp_sum': 0,
        'relation_cls_p_sum': 0.00001,
        'pred_class_tp_sum': 0,
        'pred_class_p_sum': 0.00001,
        'gt_class_tp_sum': 0,
        'gt_class_p_sum': 0.00001,
        'instance_tp_sum':0,
        'instance_p_sum': 0.00001,
        'instance_g_sum':0.00001,
        'subpred_tp_sum': 0,
        # 'subpred_p_sum': 0.00001,
        'subpred_g_sum': 0.00001,
        'predobj_tp_sum': 0,
        # 'objpred_p_sum': 0.00001,
        'predobj_g_sum': 0.00001,
        'pair_tp_sum':0,
        'pair_p_sum': 0.00001,
        'pair_g_sum':0.00001,
        'confidence_tp_sum': 0,
        'confidence_p_sum': 0.00001,
        'confidence_g_sum': 0.00001,
        'predicate_tp_sum': 0,
        'predicate_tp20_sum': 0,
        'predicate_tp50_sum': 0,
        'predicate_tp100_sum': 0,
        'predicate_p_sum': 0.00001,
        'predicate_p20_sum': 0.00001,
        'predicate_p50_sum': 0.00001,
        'predicate_p100_sum': 0.00001,
        'predicate_g_sum': 0.00001,
        'triplet_tp_sum': 0,
        'triplet_tp20_sum': 0,
        'triplet_tp50_sum': 0,
        'triplet_tp100_sum': 0,
        'triplet_p_sum': 0.00001,
        'triplet_p20_sum': 0.00001,
        'triplet_p50_sum': 0.00001,
        'triplet_p100_sum': 0.00001,
        'triplet_g_sum': 0.00001,
    }
    metrics_pr_dict = {}
    prediction_instance_json = {}
    prediction_json={}
    prediction_nopair_json={}
    object_json={}
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, len(data_loader))):
            iteration = iteration + 1
            storage.step()
            pred_instances, results_dict, losses_dict, metrics_dict = model(data, iteration, "relation",training=False)

            if 'relation_cls_tp' in metrics_dict:
                metrics_sum_dict['relation_cls_tp_sum']+=metrics_dict['relation_cls_tp']
                metrics_sum_dict['relation_cls_p_sum'] += metrics_dict['relation_cls_p']
                metrics_pr_dict['relation_cls_precision'] = metrics_sum_dict['relation_cls_tp_sum'] / metrics_sum_dict['relation_cls_p_sum']
            if 'pred_class_tp' in metrics_dict:
                metrics_sum_dict['pred_class_tp_sum']+=metrics_dict['pred_class_tp']
                metrics_sum_dict['pred_class_p_sum'] += metrics_dict['pred_class_p']
                metrics_pr_dict['pred_class_precision'] = metrics_sum_dict['pred_class_tp_sum'] / metrics_sum_dict['pred_class_p_sum']
            if 'gt_class_tp' in metrics_dict:
                metrics_sum_dict['gt_class_tp_sum']+=metrics_dict['gt_class_tp']
                metrics_sum_dict['gt_class_p_sum'] += metrics_dict['gt_class_p']
                metrics_pr_dict['gt_class_precision'] = metrics_sum_dict['gt_class_tp_sum'] / metrics_sum_dict['gt_class_p_sum']
            if 'instance_tp' in metrics_dict:
                metrics_sum_dict['instance_tp_sum']+=metrics_dict['instance_tp']
                metrics_sum_dict['instance_p_sum'] += metrics_dict['instance_p']
                metrics_sum_dict['instance_g_sum'] += metrics_dict['instance_g']
                metrics_pr_dict['instance_precision'] = metrics_sum_dict['instance_tp_sum'] / metrics_sum_dict['instance_p_sum']
                metrics_pr_dict['instance_recall'] = metrics_sum_dict['instance_tp_sum'] / metrics_sum_dict['instance_g_sum']
            if 'subpred_tp' in metrics_dict:
                metrics_sum_dict['subpred_tp_sum']+=metrics_dict['subpred_tp']
                # metrics_sum_dict['subpred_p_sum'] += metrics_dict['subpred_p']
                metrics_sum_dict['subpred_g_sum'] += metrics_dict['subpred_g']
                # metrics_pr_dict['subpred_precision'] = metrics_sum_dict['subpred_tp_sum'] / metrics_sum_dict['subpred_p_sum']
                metrics_pr_dict['subpred_recall'] = metrics_sum_dict['subpred_tp_sum'] / metrics_sum_dict['subpred_g_sum']
            if 'predobj_tp' in metrics_dict:
                metrics_sum_dict['predobj_tp_sum']+=metrics_dict['predobj_tp']
                # metrics_sum_dict['objpred_p_sum'] += metrics_dict['objpred_p']
                metrics_sum_dict['predobj_g_sum'] += metrics_dict['predobj_g']
                # metrics_pr_dict['objpred_precision'] = metrics_sum_dict['objpred_tp_sum'] / metrics_sum_dict['objpred_p_sum']
                metrics_pr_dict['predobj_recall'] = metrics_sum_dict['predobj_tp_sum'] / metrics_sum_dict['predobj_g_sum']

            if 'pair_tp' in metrics_dict:
                metrics_sum_dict['pair_tp_sum'] += metrics_dict['pair_tp']
                metrics_sum_dict['pair_p_sum'] += metrics_dict['pair_p']
                metrics_sum_dict['pair_g_sum'] += metrics_dict['pair_g']
                metrics_pr_dict['pair_precision'] = metrics_sum_dict['pair_tp_sum'] / metrics_sum_dict['pair_p_sum']
                metrics_pr_dict['pair_recall'] = metrics_sum_dict['pair_tp_sum'] / metrics_sum_dict['pair_g_sum']
            if 'confidence_tp' in metrics_dict:
                metrics_sum_dict['confidence_tp_sum']+=metrics_dict['confidence_tp']
                metrics_sum_dict['confidence_p_sum'] += metrics_dict['confidence_p']
                metrics_sum_dict['confidence_g_sum'] += metrics_dict['confidence_g']
                metrics_pr_dict['confidence_precision'] = metrics_sum_dict['confidence_tp_sum'] / metrics_sum_dict['confidence_p_sum']
                metrics_pr_dict['confidence_recall'] = metrics_sum_dict['confidence_tp_sum'] / metrics_sum_dict['confidence_g_sum']
            if 'predicate_tp' in metrics_dict:
                metrics_sum_dict['predicate_tp_sum']+=metrics_dict['predicate_tp']
                metrics_sum_dict['predicate_tp20_sum'] += metrics_dict['predicate_tp20']
                metrics_sum_dict['predicate_tp50_sum'] += metrics_dict['predicate_tp50']
                metrics_sum_dict['predicate_tp100_sum'] += metrics_dict['predicate_tp100']
                metrics_sum_dict['predicate_p_sum'] += metrics_dict['predicate_p']
                metrics_sum_dict['predicate_p20_sum'] += metrics_dict['predicate_p20']
                metrics_sum_dict['predicate_p50_sum'] += metrics_dict['predicate_p50']
                metrics_sum_dict['predicate_p100_sum'] += metrics_dict['predicate_p100']
                metrics_sum_dict['predicate_g_sum'] += metrics_dict['predicate_g']
                metrics_pr_dict['predicate_precision'] = metrics_sum_dict['predicate_tp_sum'] / metrics_sum_dict['predicate_p_sum']
                metrics_pr_dict['predicate_precision20'] = metrics_sum_dict['predicate_tp20_sum'] / metrics_sum_dict['predicate_p20_sum']
                metrics_pr_dict['predicate_precision50'] = metrics_sum_dict['predicate_tp50_sum'] / metrics_sum_dict['predicate_p50_sum']
                metrics_pr_dict['predicate_precision100'] = metrics_sum_dict['predicate_tp100_sum'] / metrics_sum_dict['predicate_p100_sum']
                metrics_pr_dict['predicate_recall'] = metrics_sum_dict['predicate_tp_sum'] / metrics_sum_dict['predicate_g_sum']
                metrics_pr_dict['predicate_recall20'] = metrics_sum_dict['predicate_tp20_sum'] / metrics_sum_dict['predicate_g_sum']
                metrics_pr_dict['predicate_recall50'] = metrics_sum_dict['predicate_tp50_sum'] / metrics_sum_dict['predicate_g_sum']
                metrics_pr_dict['predicate_recall100'] = metrics_sum_dict['predicate_tp100_sum'] / metrics_sum_dict['predicate_g_sum']
            if 'triplet_tp' in metrics_dict:
                metrics_sum_dict['triplet_tp_sum'] += metrics_dict['triplet_tp']
                metrics_sum_dict['triplet_tp20_sum'] += metrics_dict['triplet_tp20']
                metrics_sum_dict['triplet_tp50_sum'] += metrics_dict['triplet_tp50']
                metrics_sum_dict['triplet_tp100_sum'] += metrics_dict['triplet_tp100']
                metrics_sum_dict['triplet_p_sum'] += metrics_dict['triplet_p']
                metrics_sum_dict['triplet_p20_sum'] += metrics_dict['triplet_p20']
                metrics_sum_dict['triplet_p50_sum'] += metrics_dict['triplet_p50']
                metrics_sum_dict['triplet_p100_sum'] += metrics_dict['triplet_p100']
                metrics_sum_dict['triplet_g_sum'] += metrics_dict['triplet_g']
                metrics_pr_dict['triplet_precision'] = metrics_sum_dict['triplet_tp_sum'] / metrics_sum_dict['triplet_p_sum']
                metrics_pr_dict['triplet_precision20'] = metrics_sum_dict['triplet_tp20_sum'] / metrics_sum_dict['triplet_p20_sum']
                metrics_pr_dict['triplet_precision50'] = metrics_sum_dict['triplet_tp50_sum'] / metrics_sum_dict['triplet_p50_sum']
                metrics_pr_dict['triplet_precision100'] = metrics_sum_dict['triplet_tp100_sum'] / metrics_sum_dict['triplet_p100_sum']
                metrics_pr_dict['triplet_recall'] = metrics_sum_dict['triplet_tp_sum'] / metrics_sum_dict['triplet_g_sum']
                metrics_pr_dict['triplet_recall20'] = metrics_sum_dict['triplet_tp20_sum'] / metrics_sum_dict['triplet_g_sum']
                metrics_pr_dict['triplet_recall50'] = metrics_sum_dict['triplet_tp50_sum'] / metrics_sum_dict['triplet_g_sum']
                metrics_pr_dict['triplet_recall100'] = metrics_sum_dict['triplet_tp100_sum'] / metrics_sum_dict['triplet_g_sum']

            storage.put_scalars(**metrics_pr_dict, smoothing_hint=False)
            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()

            if len(pred_instances[0])>0:
                pred_boxes = pred_instances[0].pred_boxes.tensor
                height, width = pred_instances[0].image_size
                ori_height, ori_width = data[0]['height'], data[0]['width']
                pred_classes = pred_instances[0].pred_classes
                pred_boxes = torch.stack([pred_boxes[:, 1] * ori_height * 1.0 / height,
                                          pred_boxes[:, 0] * ori_width * 1.0 / width,
                                          pred_boxes[:, 3] * ori_height * 1.0 / height,
                                          pred_boxes[:, 2] * ori_width * 1.0 / width], dim=1)

                pred_classes = pred_classes.data.cpu().numpy()
                pred_boxes = pred_boxes.data.cpu().numpy()
                # pred_masks = pred_instances[0].pred_masks.data.cpu().numpy()
                # print(pred_masks.shape)
                # pred_masks_encode = []
                # for mask in pred_masks:
                #     mask_encode=maskUtils.encode(cv2.resize(np.asfortranarray(mask),(width,height),cv2.INTER_NEAREST))
                #     pred_masks_encode.append({'size':mask_encode['size'],'counts':mask_encode['counts'].decode()})

                ## triplet as output
                if 'triplet_interest_pred' in results_dict:
                    predicate_categories = results_dict['predicate_categories'][0].data.cpu().numpy().reshape(len(pred_instances[0]), len(pred_instances[0]), cfg.MODEL.PREDICATE_HEADS.RELATION_NUM-1)
                    triplet_interest_pred = results_dict['triplet_interest_pred'][0].data.cpu().numpy().reshape(len(pred_instances[0]), len(pred_instances[0]), cfg.MODEL.PREDICATE_HEADS.RELATION_NUM-1)

                    pair_interest_pred = results_dict['pair_interest_pred'][0].data.cpu().numpy().reshape(len(pred_instances[0]), len(pred_instances[0]))
                    pair_interest_pred_instance_pair = pair_interest_pred * (1 - np.eye(len(pred_instances[0])))
                    predicate_factor = pair_interest_pred_instance_pair.reshape(len(pred_instances[0]),len(pred_instances[0]), 1)
                    single_result = (predicate_factor * predicate_categories * triplet_interest_pred).reshape(-1)
                    single_result_indx = np.argsort(single_result)[::-1][:100]
                    single_index = []
                    for i in range(len(pred_instances[0])):
                        for j in range(len(pred_instances[0])):
                            for k in range(cfg.MODEL.PREDICATE_HEADS.RELATION_NUM-1):
                                single_index.append([i, j, k])
                    single_index = np.array(single_index)
                    locations = single_index[single_result_indx]
                    scores = single_result[single_result_indx]
                    prediction_json[str(data[0]['image_id'])] = {
                        "relation_ids": (locations[:, 2] + 1).tolist(),
                        "subject_class_ids": pred_classes[locations[:, 0]].tolist(),
                        "subject_boxes": pred_boxes[locations[:, 0]].tolist(),
                        "object_class_ids": pred_classes[locations[:, 1]].tolist(),
                        "object_boxes": pred_boxes[locations[:, 1]].tolist(),
                        "scores": scores.tolist()
                    }

                    pair_interest_pred_instance_nopair = 1 - np.eye(len(pred_instances[0]))
                    predicate_factor = pair_interest_pred_instance_nopair.reshape(len(pred_instances[0]),len(pred_instances[0]), 1)

                    single_result_nopair = (predicate_factor*triplet_interest_pred).reshape(-1)
                    single_result_indx_nopair = np.argsort(single_result_nopair)[::-1][:100]
                    single_index_nopair = []
                    for i in range(len(pred_instances[0])):
                        for j in range(len(pred_instances[0])):
                            for k in range(cfg.MODEL.PREDICATE_HEADS.RELATION_NUM-1):
                                single_index_nopair.append([i, j, k])
                    single_index_nopair = np.array(single_index_nopair)
                    locations_nopair = single_index_nopair[single_result_indx_nopair]
                    scores_nopair = single_result_nopair[single_result_indx_nopair]
                    prediction_nopair_json[str(data[0]['image_id'])] = {
                        "relation_ids": (locations_nopair[:, 2] + 1).tolist(),
                        "subject_class_ids": pred_classes[locations_nopair[:, 0]].tolist(),
                        "subject_boxes": pred_boxes[locations_nopair[:, 0]].tolist(),
                        "object_class_ids": pred_classes[locations_nopair[:, 1]].tolist(),
                        "object_boxes": pred_boxes[locations_nopair[:, 1]].tolist(),
                        "scores": scores_nopair.tolist()
                    }
                ## only raw predicate
                elif 'pair_interest_pred' not in results_dict:
                    object_json[str(data[0]['image_id'])] = {
                        "class_ids": pred_classes.tolist(),
                        "boxes": pred_boxes.tolist(),
                        # "masks": pred_masks_encode,
                        # "scores": []
                    }
                    single_result = results_dict['predicate_categories'][0].data.cpu().numpy().reshape(len(pred_instances[0]), len(pred_instances[0]), cfg.MODEL.PREDICATE_HEADS.RELATION_NUM-1).reshape(-1)
                    single_result_indx = np.argsort(single_result)[::-1][:100]
                    single_index = []
                    for i in range(len(pred_instances[0])):
                        for j in range(len(pred_instances[0])):
                            for k in range(cfg.MODEL.PREDICATE_HEADS.RELATION_NUM-1):
                                single_index.append([i, j, k])
                    single_index = np.array(single_index)
                    locations = single_index[single_result_indx]
                    scores = single_result[single_result_indx]

                    prediction_json[str(data[0]['image_id'])] = {
                        "locations": locations.tolist(),
                        "relation_ids": (locations[:, 2] + 1).tolist(),
                        "subject_class_ids": pred_classes[locations[:, 0]].tolist(),
                        "subject_boxes": pred_boxes[locations[:, 0]].tolist(),
                        "object_class_ids": pred_classes[locations[:, 1]].tolist(),
                        "object_boxes": pred_boxes[locations[:, 1]].tolist(),
                        "scores": scores.tolist()
                    }
                    prediction_nopair_json[str(data[0]['image_id'])] = {
                        "locations": locations.tolist(),
                        "relation_ids": (locations[:, 2] + 1).tolist(),
                        "subject_class_ids": pred_classes[locations[:, 0]].tolist(),
                        "subject_boxes": pred_boxes[locations[:, 0]].tolist(),
                        "object_class_ids": pred_classes[locations[:, 1]].tolist(),
                        "object_boxes": pred_boxes[locations[:, 1]].tolist(),
                        "scores": scores.tolist()
                    }
                else:
                    object_json[str(data[0]['image_id'])] = {
                        "class_ids": pred_classes.tolist(),
                        "boxes": pred_boxes.tolist(),
                        # "masks": pred_masks_encode,
                        # "scores": []
                    }
                    predicate_categories = results_dict['predicate_categories'][0].data.cpu().numpy().reshape(len(pred_instances[0]), len(pred_instances[0]), cfg.MODEL.PREDICATE_HEADS.RELATION_NUM - 1)
                    instance_interest_pred = results_dict['instance_interest_pred'][0]
                    sub_instance_interest_pred = instance_interest_pred.view(-1,1).expand(len(pred_instances[0]),len(pred_instances[0])).data.cpu().numpy()
                    obj_instance_interest_pred = instance_interest_pred.view(1, -1).expand(len(pred_instances[0]),len(pred_instances[0])).data.cpu().numpy()
                    pair_interest_pred = results_dict['pair_interest_pred'][0].data.cpu().numpy().reshape(len(pred_instances[0]), len(pred_instances[0]))

                    pair_interest_pred_instance_pair_instance = pair_interest_pred * (1 - np.eye(len(pred_instances[0]))) * sub_instance_interest_pred * obj_instance_interest_pred
                    predicate_factor_instance = pair_interest_pred_instance_pair_instance.reshape(len(pred_instances[0]),len(pred_instances[0]), 1)
                    single_result_instance = (predicate_factor_instance * predicate_categories).reshape(-1)
                    single_result_indx_instance = np.argsort(single_result_instance)[::-1][:100]
                    single_index_instance = []
                    for i in range(len(pred_instances[0])):
                        for j in range(len(pred_instances[0])):
                            for k in range(cfg.MODEL.PREDICATE_HEADS.RELATION_NUM - 1):
                                single_index_instance.append([i, j, k])
                    single_index_instance = np.array(single_index_instance)
                    locations_instance = single_index_instance[single_result_indx_instance]
                    scores_instance = single_result_instance[single_result_indx_instance]

                    prediction_instance_json[str(data[0]['image_id'])] = {
                        "locations": locations_instance.tolist(),
                        "relation_ids": (locations_instance[:, 2] + 1).tolist(),
                        "subject_class_ids": pred_classes[locations_instance[:, 0]].tolist(),
                        "subject_boxes": pred_boxes[locations_instance[:, 0]].tolist(),
                        "object_class_ids": pred_classes[locations_instance[:, 1]].tolist(),
                        "object_boxes": pred_boxes[locations_instance[:, 1]].tolist(),
                        "scores": scores_instance.tolist()
                    }

                    pair_interest_pred_instance_pair = pair_interest_pred * (1 - np.eye(len(pred_instances[0])))
                    predicate_factor = pair_interest_pred_instance_pair.reshape(len(pred_instances[0]), len(pred_instances[0]), 1)
                    single_result = (predicate_factor * predicate_categories).reshape(-1)

                    single_result_indx = np.argsort(single_result)[::-1][:100]
                    single_index = []
                    for i in range(len(pred_instances[0])):
                        for j in range(len(pred_instances[0])):
                            for k in range(cfg.MODEL.PREDICATE_HEADS.RELATION_NUM-1):
                                single_index.append([i, j, k])
                    single_index = np.array(single_index)
                    locations = single_index[single_result_indx]
                    scores = single_result[single_result_indx]

                    prediction_json[str(data[0]['image_id'])] = {
                        "locations":locations.tolist(),
                        "relation_ids": (locations[:, 2] + 1).tolist(),
                        "subject_class_ids": pred_classes[locations[:, 0]].tolist(),
                        "subject_boxes": pred_boxes[locations[:, 0]].tolist(),
                        "object_class_ids": pred_classes[locations[:, 1]].tolist(),
                        "object_boxes": pred_boxes[locations[:, 1]].tolist(),
                        "scores": scores.tolist()
                    }

                    pair_interest_pred_instance_pair_nopair = 1 - np.eye(len(pred_instances[0]))
                    predicate_factor_nopair = pair_interest_pred_instance_pair_nopair.reshape(len(pred_instances[0]),len(pred_instances[0]), 1)
                    single_result_nopair = (predicate_factor_nopair * predicate_categories).reshape(-1)
                    single_result_indx_nopair = np.argsort(single_result_nopair)[::-1][:100]
                    single_index_nopair = []
                    for i in range(len(pred_instances[0])):
                        for j in range(len(pred_instances[0])):
                            for k in range(cfg.MODEL.PREDICATE_HEADS.RELATION_NUM-1):
                                single_index_nopair.append([i, j, k])
                    single_index_nopair = np.array(single_index_nopair)
                    locations_nopair = single_index_nopair[single_result_indx_nopair]
                    scores_nopair = single_result_nopair[single_result_indx_nopair]
                    prediction_nopair_json[str(data[0]['image_id'])] = {
                        "locations": locations_nopair.tolist(),
                        "relation_ids": (locations_nopair[:, 2] + 1).tolist(),
                        "subject_class_ids": pred_classes[locations_nopair[:, 0]].tolist(),
                        "subject_boxes": pred_boxes[locations_nopair[:, 0]].tolist(),
                        "object_class_ids": pred_classes[locations_nopair[:, 1]].tolist(),
                        "object_boxes": pred_boxes[locations_nopair[:, 1]].tolist(),
                        "scores": scores_nopair.tolist()
                    }
            else:
                object_json[str(data[0]['image_id'])]={
                    "class_ids":[],
                    "boxes":[],
                    # "masks":[],
                    # "scores":[]
                }
                prediction_instance_json[str(data[0]['image_id'])] = {
                    "locations": [],
                    "relation_ids": [],
                    "subject_class_ids": [],
                    "subject_boxes": [],
                    "object_class_ids": [],
                    "object_boxes": [],
                    "scores": []
                }
                prediction_json[str(data[0]['image_id'])] = {
                    "locations": [],
                    "relation_ids": [],
                    "subject_class_ids": [],
                    "subject_boxes": [],
                    "object_class_ids": [],
                    "object_boxes": [],
                    "scores": []
                }
                prediction_nopair_json[str(data[0]['image_id'])] = {
                    "locations": [],
                    "relation_ids": [],
                    "subject_class_ids": [],
                    "subject_boxes": [],
                    "object_class_ids": [],
                    "object_boxes": [],
                    "scores": []
                }
            # torch.cuda.empty_cache()
            # break
    return object_json,prediction_instance_json,prediction_json,prediction_nopair_json

def do_relation_train(cfg, model, resume=False):
    model.train()
    for param in model.named_parameters():
        param[1].requires_grad = False
    for param in model.named_parameters():
        for trainable in cfg.MODEL.TRAINABLE:
            if param[0].startswith(trainable):
                param[1].requires_grad = True
                break

        if param[0] == "relation_heads.instance_head.semantic_embed.weight" or \
            param[0] == "relation_heads.pair_head.semantic_embed.weight" or \
            param[0] == "relation_heads.predicate_head.semantic_embed.weight" or \
            param[0] == "relation_heads.triplet_head.ins_embed.weight" or \
            param[0] == "relation_heads.triplet_head.pred_embed.weight" or \
            param[0] == "relation_heads.subpred_head.sub_embed.weight" or \
            param[0] == "relation_heads.subpred_head.pred_embed.weight" or \
            param[0] == "relation_heads.predobj_head.pred_embed.weight" or \
            param[0] == "relation_heads.predobj_head.obj_embed.weight" or \
            param[0].startswith("relation_heads.predicate_head.freq_bias.obj_baseline.weight"):
            param[1].requires_grad = False

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    metrics_sum_dict = {
        'relation_cls_tp_sum': 0,
        'relation_cls_p_sum': 0.00001,
        'pred_class_tp_sum': 0,
        'pred_class_p_sum': 0.00001,
        'gt_class_tp_sum': 0,
        'gt_class_p_sum': 0.00001,
        'raw_pred_class_tp_sum': 0,
        'raw_pred_class_p_sum': 0.00001,
        'instance_tp_sum':0,
        'instance_p_sum': 0.00001,
        'instance_g_sum':0.00001,
        'subpred_tp_sum': 0,
        'subpred_p_sum': 0.00001,
        'subpred_g_sum': 0.00001,
        'predobj_tp_sum': 0,
        'predobj_p_sum': 0.00001,
        'predobj_g_sum': 0.00001,
        'pair_tp_sum':0,
        'pair_p_sum': 0.00001,
        'pair_g_sum':0.00001,
        'confidence_tp_sum': 0,
        'confidence_p_sum': 0.00001,
        'confidence_g_sum': 0.00001,
        'predicate_tp_sum': 0,
        'predicate_tp20_sum': 0,
        'predicate_tp50_sum': 0,
        'predicate_tp100_sum': 0,
        'predicate_p_sum': 0.00001,
        'predicate_p20_sum': 0.00001,
        'predicate_p50_sum': 0.00001,
        'predicate_p100_sum': 0.00001,
        'predicate_g_sum': 0.00001,
        'triplet_tp_sum': 0,
        'triplet_tp20_sum': 0,
        'triplet_tp50_sum': 0,
        'triplet_tp100_sum': 0,
        'triplet_p_sum': 0.00001,
        'triplet_p20_sum': 0.00001,
        'triplet_p50_sum': 0.00001,
        'triplet_p100_sum': 0.00001,
        'triplet_g_sum': 0.00001,
    }
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler, metrics_sum_dict=metrics_sum_dict
    )
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    # state_dict=torch.load(cfg.MODEL.WEIGHTS).pop("model")
    # model.load_state_dict(state_dict,strict=False)
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    # relation_cls_state_dict=torch.load(cfg.MODEL.WEIGHTS).pop("model")
    # for param in model.named_parameters():
    #     if param[0] not in relation_cls_state_dict:
    #         print(param[0])
    # model.load_state_dict(relation_cls_state_dict,strict=False)

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    metrics_pr_dict={}
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    acumulate_losses=0
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            print(iteration)
            iteration = iteration + 1
            storage.step()
            if True:
            # try:
                pred_instances, results_dict, losses_dict, metrics_dict = model(data,iteration,mode="relation",training=True)
                losses = sum(loss for loss in losses_dict.values())
                assert torch.isfinite(losses).all(), losses_dict
                #print(losses_dict)

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(losses_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                acumulate_losses += losses_reduced
                if comm.is_main_process():
                    storage.put_scalars(acumulate_losses=acumulate_losses/(iteration-start_iter),total_loss=losses_reduced, **loss_dict_reduced)

                if 'relation_cls_tp' in metrics_dict:
                    metrics_sum_dict['relation_cls_tp_sum']+=metrics_dict['relation_cls_tp']
                    metrics_sum_dict['relation_cls_p_sum'] += metrics_dict['relation_cls_p']
                    metrics_pr_dict['relation_cls_precision'] = metrics_sum_dict['relation_cls_tp_sum'] / metrics_sum_dict['relation_cls_p_sum']
                if 'pred_class_tp' in metrics_dict:
                    metrics_sum_dict['pred_class_tp_sum']+=metrics_dict['pred_class_tp']
                    metrics_sum_dict['pred_class_p_sum'] += metrics_dict['pred_class_p']
                    metrics_pr_dict['pred_class_precision'] = metrics_sum_dict['pred_class_tp_sum'] / metrics_sum_dict['pred_class_p_sum']
                if 'raw_pred_class_tp' in metrics_dict:
                    metrics_sum_dict['raw_pred_class_tp_sum']+=metrics_dict['raw_pred_class_tp']
                    metrics_sum_dict['raw_pred_class_p_sum'] += metrics_dict['raw_pred_class_p']
                    metrics_pr_dict['raw_pred_class_precision'] = metrics_sum_dict['raw_pred_class_tp_sum'] / metrics_sum_dict['raw_pred_class_p_sum']
                if 'gt_class_tp' in metrics_dict:
                    metrics_sum_dict['gt_class_tp_sum']+=metrics_dict['gt_class_tp']
                    metrics_sum_dict['gt_class_p_sum'] += metrics_dict['gt_class_p']
                    metrics_pr_dict['gt_class_precision'] = metrics_sum_dict['gt_class_tp_sum'] / metrics_sum_dict['gt_class_p_sum']
                if 'instance_tp' in metrics_dict:
                    metrics_sum_dict['instance_tp_sum']+=metrics_dict['instance_tp']
                    metrics_sum_dict['instance_p_sum'] += metrics_dict['instance_p']
                    metrics_sum_dict['instance_g_sum'] += metrics_dict['instance_g']
                    metrics_pr_dict['instance_precision'] = metrics_sum_dict['instance_tp_sum'] / metrics_sum_dict['instance_p_sum']
                    metrics_pr_dict['instance_recall'] = metrics_sum_dict['instance_tp_sum'] / metrics_sum_dict['instance_g_sum']
                if 'subpred_tp' in metrics_dict:
                    metrics_sum_dict['subpred_tp_sum']+=metrics_dict['subpred_tp']
                    metrics_sum_dict['subpred_p_sum'] += metrics_dict['subpred_p']
                    metrics_sum_dict['subpred_g_sum'] += metrics_dict['subpred_g']
                    metrics_pr_dict['subpred_precision'] = metrics_sum_dict['subpred_tp_sum'] / metrics_sum_dict['subpred_p_sum']
                    metrics_pr_dict['subpred_recall'] = metrics_sum_dict['subpred_tp_sum'] / metrics_sum_dict['subpred_g_sum']
                if 'predobj_tp' in metrics_dict:
                    metrics_sum_dict['predobj_tp_sum']+=metrics_dict['predobj_tp']
                    metrics_sum_dict['predobj_p_sum'] += metrics_dict['predobj_p']
                    metrics_sum_dict['predobj_g_sum'] += metrics_dict['predobj_g']
                    metrics_pr_dict['predobj_precision'] = metrics_sum_dict['predobj_tp_sum'] / metrics_sum_dict['predobj_p_sum']
                    metrics_pr_dict['predobj_recall'] = metrics_sum_dict['predobj_tp_sum'] / metrics_sum_dict['predobj_g_sum']

                if 'pair_tp' in metrics_dict:
                    metrics_sum_dict['pair_tp_sum'] += metrics_dict['pair_tp']
                    metrics_sum_dict['pair_p_sum'] += metrics_dict['pair_p']
                    metrics_sum_dict['pair_g_sum'] += metrics_dict['pair_g']
                    metrics_pr_dict['pair_precision'] = metrics_sum_dict['pair_tp_sum'] / metrics_sum_dict['pair_p_sum']
                    metrics_pr_dict['pair_recall'] = metrics_sum_dict['pair_tp_sum'] / metrics_sum_dict['pair_g_sum']
                if 'confidence_tp' in metrics_dict:
                    metrics_sum_dict['confidence_tp_sum']+=metrics_dict['confidence_tp']
                    metrics_sum_dict['confidence_p_sum'] += metrics_dict['confidence_p']
                    metrics_sum_dict['confidence_g_sum'] += metrics_dict['confidence_g']
                    metrics_pr_dict['confidence_precision'] = metrics_sum_dict['confidence_tp_sum'] / metrics_sum_dict['confidence_p_sum']
                    metrics_pr_dict['confidence_recall'] = metrics_sum_dict['confidence_tp_sum'] / metrics_sum_dict['confidence_g_sum']
                if 'predicate_tp' in metrics_dict:
                    metrics_sum_dict['predicate_tp_sum']+=metrics_dict['predicate_tp']
                    metrics_sum_dict['predicate_tp20_sum'] += metrics_dict['predicate_tp20']
                    metrics_sum_dict['predicate_tp50_sum'] += metrics_dict['predicate_tp50']
                    metrics_sum_dict['predicate_tp100_sum'] += metrics_dict['predicate_tp100']
                    metrics_sum_dict['predicate_p_sum'] += metrics_dict['predicate_p']
                    metrics_sum_dict['predicate_p20_sum'] += metrics_dict['predicate_p20']
                    metrics_sum_dict['predicate_p50_sum'] += metrics_dict['predicate_p50']
                    metrics_sum_dict['predicate_p100_sum'] += metrics_dict['predicate_p100']
                    metrics_sum_dict['predicate_g_sum'] += metrics_dict['predicate_g']
                    metrics_pr_dict['predicate_precision'] = metrics_sum_dict['predicate_tp_sum'] / metrics_sum_dict['predicate_p_sum']
                    metrics_pr_dict['predicate_precision20'] = metrics_sum_dict['predicate_tp20_sum'] / metrics_sum_dict['predicate_p20_sum']
                    metrics_pr_dict['predicate_precision50'] = metrics_sum_dict['predicate_tp50_sum'] / metrics_sum_dict['predicate_p50_sum']
                    metrics_pr_dict['predicate_precision100'] = metrics_sum_dict['predicate_tp100_sum'] / metrics_sum_dict['predicate_p100_sum']
                    metrics_pr_dict['predicate_recall'] = metrics_sum_dict['predicate_tp_sum'] / metrics_sum_dict['predicate_g_sum']
                    metrics_pr_dict['predicate_recall20'] = metrics_sum_dict['predicate_tp20_sum'] / metrics_sum_dict['predicate_g_sum']
                    metrics_pr_dict['predicate_recall50'] = metrics_sum_dict['predicate_tp50_sum'] / metrics_sum_dict['predicate_g_sum']
                    metrics_pr_dict['predicate_recall100'] = metrics_sum_dict['predicate_tp100_sum'] / metrics_sum_dict['predicate_g_sum']
                if 'triplet_tp' in metrics_dict:
                    metrics_sum_dict['triplet_tp_sum'] += metrics_dict['triplet_tp']
                    metrics_sum_dict['triplet_tp20_sum'] += metrics_dict['triplet_tp20']
                    metrics_sum_dict['triplet_tp50_sum'] += metrics_dict['triplet_tp50']
                    metrics_sum_dict['triplet_tp100_sum'] += metrics_dict['triplet_tp100']
                    metrics_sum_dict['triplet_p_sum'] += metrics_dict['triplet_p']
                    metrics_sum_dict['triplet_p20_sum'] += metrics_dict['triplet_p20']
                    metrics_sum_dict['triplet_p50_sum'] += metrics_dict['triplet_p50']
                    metrics_sum_dict['triplet_p100_sum'] += metrics_dict['triplet_p100']
                    metrics_sum_dict['triplet_g_sum'] += metrics_dict['triplet_g']
                    metrics_pr_dict['triplet_precision'] = metrics_sum_dict['triplet_tp_sum'] / metrics_sum_dict['triplet_p_sum']
                    metrics_pr_dict['triplet_precision20'] = metrics_sum_dict['triplet_tp20_sum'] / metrics_sum_dict['triplet_p20_sum']
                    metrics_pr_dict['triplet_precision50'] = metrics_sum_dict['triplet_tp50_sum'] / metrics_sum_dict['triplet_p50_sum']
                    metrics_pr_dict['triplet_precision100'] = metrics_sum_dict['triplet_tp100_sum'] / metrics_sum_dict['triplet_p100_sum']
                    metrics_pr_dict['triplet_recall'] = metrics_sum_dict['triplet_tp_sum'] / metrics_sum_dict['triplet_g_sum']
                    metrics_pr_dict['triplet_recall20'] = metrics_sum_dict['triplet_tp20_sum'] / metrics_sum_dict['triplet_g_sum']
                    metrics_pr_dict['triplet_recall50'] = metrics_sum_dict['triplet_tp50_sum'] / metrics_sum_dict['triplet_g_sum']
                    metrics_pr_dict['triplet_recall100'] = metrics_sum_dict['triplet_tp100_sum'] / metrics_sum_dict['triplet_g_sum']

                storage.put_scalars(**metrics_pr_dict, smoothing_hint=False)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)
                torch.cuda.empty_cache()
            # except Exception as e:
            #     print(e)

def do_relation_test_demo(cfg,image_path,visible=False,visible_num=5):
    predictor = DefaultPredictor(cfg)
    img = read_image(image_path, format="BGR")
    pred_instances, results_dict = predictor(img, 0, mode="relation")
    image_info={}

    if len(pred_instances[0]) > 0:
        pred_boxes = pred_instances[0].pred_boxes.tensor
        height, width = pred_instances[0].image_size
        ori_height, ori_width = img.shape[0], img.shape[1]
        pred_classes = pred_instances[0].pred_classes
        pred_boxes = torch.stack([pred_boxes[:, 1] * ori_height * 1.0 / height,
                                  pred_boxes[:, 0] * ori_width * 1.0 / width,
                                  pred_boxes[:, 3] * ori_height * 1.0 / height,
                                  pred_boxes[:, 2] * ori_width * 1.0 / width], dim=1)

        pred_classes = pred_classes.data.cpu().numpy()
        pred_boxes = pred_boxes.data.cpu().numpy()

        predicate_categories = results_dict['predicate_categories'][0].data.cpu().numpy().reshape(
            len(pred_instances[0]), len(pred_instances[0]), 249)

        pair_interest_pred = results_dict['pair_interest_pred'][0].data.cpu().numpy().reshape(len(pred_instances[0]),
                                                                                              len(pred_instances[0]))
        pair_interest_pred_instance_pair = pair_interest_pred * (1 - np.eye(len(pred_instances[0])))
        predicate_factor = pair_interest_pred_instance_pair.reshape(len(pred_instances[0]), len(pred_instances[0]), 1)
        single_result = (predicate_factor * predicate_categories).reshape(-1)
        single_result_indx = np.argsort(single_result)[::-1][:100]
        single_index = []
        for i in range(len(pred_instances[0])):
            for j in range(len(pred_instances[0])):
                for k in range(249):
                    single_index.append([i, j, k])
        single_index = np.array(single_index)
        locations = single_index[single_result_indx]
        scores = single_result[single_result_indx]
        image_info[image_path] = {
            "relation_ids": (locations[:, 2] + 1).tolist(),
            "subject_class_ids": pred_classes[locations[:, 0]].tolist(),
            "subject_boxes": pred_boxes[locations[:, 0]].tolist(),
            "object_class_ids": pred_classes[locations[:, 1]].tolist(),
            "object_boxes": pred_boxes[locations[:, 1]].tolist(),
            "scores": scores.tolist()
        }
    else:
        image_info[image_path] = {
            "relation_ids": [],
            "subject_class_ids": [],
            "subject_boxes": [],
            "object_class_ids": [],
            "object_boxes": [],
            "scores": []
        }

    if visible:
        subject_boxes = image_info[image_path]['subject_boxes']
        object_boxes = image_info[image_path]['object_boxes']
        subject_boxes_xyxy = []
        object_boxes_xyxy = []
        for sub_box, obj_box in zip(subject_boxes, object_boxes):
            subject_boxes_xyxy.append([sub_box[1], sub_box[0], sub_box[3], sub_box[2]])
            object_boxes_xyxy.append([obj_box[1], obj_box[0], obj_box[3], obj_box[2]])
        subject_class_ids = image_info[image_path]['subject_class_ids']
        object_class_ids = image_info[image_path]['object_class_ids']
        scores = image_info[image_path]['scores']
        relation_ids = image_info[image_path]['relation_ids']

        subject_boxes = np.array(subject_boxes_xyxy)
        object_boxes = np.array(object_boxes_xyxy)
        subject_class_ids = np.array(subject_class_ids)
        object_class_ids = np.array(object_class_ids)
        relation_ids = np.array(relation_ids)
        scores = np.array(scores)

        sort_idx = np.argsort(-scores)[:visible_num]
        triplets = Instances((img.shape[0], img.shape[1]))
        triplets.subject_classes = torch.Tensor(subject_class_ids[sort_idx]).int() - 1
        triplets.object_classes = torch.Tensor(object_class_ids[sort_idx]).int() - 1
        triplets.subject_boxes = Boxes(torch.Tensor(subject_boxes[sort_idx, :]))
        triplets.object_boxes = Boxes(torch.Tensor(object_boxes[sort_idx, :]))
        triplets.relations = torch.Tensor(relation_ids[sort_idx]).int() - 1
        triplets.scores = torch.Tensor(scores[sort_idx])
        visualizer = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TEST[0]), instance_mode=ColorMode.IMAGE)
        vis_output_instance = visualizer.draw_relation_predictions(triplets)
        vis_output_instance.save(os.path.join(image_path.split("/")[-1].split(".")[0] + "_" + str(visible_num) + ".png"))
    return image_info

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    print(cfg.MODEL.DEVICE)
    model.eval()
    logger.info("Model:\n{}".format(model))
    if args.mode=="test_panoptic":
        do_panoptic_test(cfg,model)
    elif args.mode=="train_relation":
        do_relation_train(cfg, model)
    elif args.mode=="test_relation":
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        instances,prediction_instance_json,prediction_json,prediction_nopair_json=do_relation_test(cfg, model)
        # json.dump(instances, open("./output/" + args.config_file.split("/")[-1] + "_final_instances.json", 'w'))
        json.dump(prediction_instance_json, open("./output/" + args.config_file.split("/")[-1] + "_instance.json", 'w'))
        json.dump(prediction_json,open("./output/"+args.config_file.split("/")[-1]+".json",'w'))
        json.dump(prediction_nopair_json,open("./output/"+args.config_file.split("/")[-1]+"_nopair.json",'w'))
    elif args.mode=="demo":
        do_relation_test_demo(cfg, args.image_path, args.visible, args.visible_num)
    else:
        print("mode not supported")

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
