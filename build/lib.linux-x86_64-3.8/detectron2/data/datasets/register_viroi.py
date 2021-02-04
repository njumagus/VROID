# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy

from detectron2.data import DatasetCatalog, MetadataCatalog
from .viroi import load_viroi_json

"""
This file contains functions to register a COCO-format dataset to the DatasetCatalog.
"""

__all__ = ["register_viroi"]

def register_viroi(name, metadata,
                image_path,
                stuff_path,
                panoptic_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_viroi_json(name,
                image_path,
                stuff_path,
                panoptic_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_path=image_path,
        stuff_path=stuff_path,
        panoptic_path=panoptic_path,
        class_json_file=class_json_file,
        relation_json_file=relation_json_file,
        instance_json_file=instance_json_file,
        triplet_json_file=triplet_json_file, evaluator_type="viroi", **metadata
    )
