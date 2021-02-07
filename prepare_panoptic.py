# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time
import functools
import json
import multiprocessing as mp
import numpy as np
import os
from PIL import Image

from panopticapi.utils import rgb2id


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for instance_id in segments:
        seg = segments[instance_id]
        cat_id = seg["class_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["instance_id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, class_dict):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.

    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    stuff_ids = [int(k) for k in class_dict if class_dict[k]["isthing"] == 0]
    thing_ids = [int(k) for k in class_dict if class_dict[k]["isthing"] == 1]
    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(stuff_ids) <= 254
    for i, stuff_id in enumerate(stuff_ids):
        id_map[stuff_id] = i + 1
    for thing_id in thing_ids:
        id_map[thing_id] = 0
    id_map[0] = 255

    with open(panoptic_json) as f:
        images_dict = json.load(f)

    # pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    # def iter_annotations():
    #     for image_id in images_dict:
    #         image_dict=images_dict[image_id]
    #         file_name = image_dict["image_name"]
    #         segments = image_dict["instances"]
    #         input = os.path.join(panoptic_root, file_name.replace("jpg","png"))
    #         output = os.path.join(sem_seg_root, file_name.replace("jpg","png"))
    #         yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    exists = os.listdir(sem_seg_root)
    for image_id in images_dict:
        image_dict = images_dict[image_id]
        file_name = image_dict["image_name"]
        if file_name.replace("jpg", "png") not in exists:
            try:
                segments = image_dict["instances"]
                input = os.path.join(panoptic_root, file_name.replace("jpg", "png"))
                output = os.path.join(sem_seg_root, file_name.replace("jpg", "png"))
                _process_panoptic_to_semantic(input, output, segments, id_map)
            except:
                print(file_name)

    # pool.starmap(
    #     functools.partial(_process_panoptic_to_semantic, id_map=id_map),
    #     iter_annotations(),
    #     chunksize=100,
    # )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    class_dict = json.load(open("../data/viroi_json/class_dict.json", 'r'))
    for s in ["train", "test"]:
        separate_coco_semantic_from_panoptic(
            os.path.join("../data/viroi_json", "{}_images_dict.json".format(s)),
            "../data/ioid_panoptic",
            "../data/viroi_stuff",
            class_dict,
        )
