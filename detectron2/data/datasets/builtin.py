# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from .register_viroi import register_viroi
from .builtin_meta import _get_builtin_metadata


# ==== Predefined datasets and splits for COCO ==========
_PREDEFINED_SPLITS_REOID = {}
_PREDEFINED_SPLITS_REOID["viroi"] = {
    "viroi_train": ("../data/ioid_images","../data/ioid_stuff","../data/ioid_panoptic","../data/viroi_json/class_dict.json","../data/viroi_json/relation_dict.json","../data/viroi_json/train_images_dict.json","../data/viroi_json/train_images_triplets_dict.json"),
    "viroi_test": ("../data/ioid_images","../data/ioid_stuff","../data/ioid_panoptic","../data/viroi_json/class_dict.json","../data/viroi_json/relation_dict.json","../data/viroi_json/test_images_dict.json","../data/viroi_json/test_images_triplets_dict.json"),
    "viroi_test5": ("../data/ioid_images","../data/ioid_stuff","../data/ioid_panoptic","../data/viroi_json/class_dict.json","../data/viroi_json/relation_dict.json","../data/viroi_json/5_images_dict.json","../data/viroi_json/5_images_triplets_dict.json"),

    "viroi_minitrain": ("../data/ioid_images","../data/ioid_stuff","../data/ioid_panoptic","../data/viroi_json/class_dict.json","../data/viroi_json/relation_dict.json","../data/viroi_json/mini/minitrain_images_dict.json","../data/viroi_json/mini/mini_train_images_triplets_dict.json"),
    "viroi_minival": ("../data/ioid_images","../data/ioid_stuff","../data/ioid_panoptic","../data/viroi_json/class_dict.json","../data/viroi_json/relation_dict.json","../data/viroi_json/mini/mini_test_images_dict.json","../data/viroi_json/mini/mini_test_images_triplets_dict.json")
}

def register_all_viroi(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_REOID.items():
        for key, (image_path,stuff_path,panoptic_path,class_json_file,relation_json_file,instance_json_file,triplet_json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_viroi(
                key,
                _get_builtin_metadata(dataset_name),
                image_path,
                stuff_path,
                panoptic_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file
            )

register_all_viroi()