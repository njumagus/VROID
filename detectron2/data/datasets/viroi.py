# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock
import pycocotools.mask as mask_utils

from detectron2.data import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_viroi_json"]

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class VIROI:
    def __init__(self,image_path,stuff_path,panoptic_path,class_json_file,relation_json_file,instance_json_file,triplet_json_file):
        self.image_path=image_path
        self.stuff_path=stuff_path
        self.panoptic_path = panoptic_path
        self.class_dict=json.load(open(class_json_file))
        self.relation_dict=json.load(open(relation_json_file))
        self.image_instance_dict=json.load(open(instance_json_file))
        self.image_triplet_dict = json.load(open(triplet_json_file))

        self.thing_list = []
        self.stuff_list = []
        for class_id in range(1,len(self.class_dict)+1): # from 1 to 133
            if self.class_dict[str(class_id)]['isthing']==1:
                self.thing_list.append(self.class_dict[str(class_id)]) # list of thing dicts from 1 to 80
            else:
                self.stuff_list.append(self.class_dict[str(class_id)]) # list of stuff dicts from 81 to 133

        self.image_id_list=sorted([int(image_id) for image_id in list(self.image_instance_dict.keys())])

    def loadClassdict(self):
        return self.class_dict

    def loadThings(self):
        return self.thing_list

    def loadStuffs(self):
        return self.stuff_list

    def loadIds(self):
        return self.image_id_list

    def loadImgs(self,ids):
        if _isArrayLike(ids):
            return [self.image_instance_dict[str(id)] for id in ids]
        elif type(ids) == int:
            return [self.image_instance_dict[str(ids)]]

    def loadInstances(self,ids):
        if _isArrayLike(ids):
            return [self.image_instance_dict[str(id)]['instances'] for id in ids]
        elif type(ids) == int:
            return [self.image_instance_dict[str(ids)]['instances']]

    def loadTriplets(self,ids):
        if _isArrayLike(ids):
            return [self.image_triplet_dict[str(id)]['triplets'] for id in ids]
        elif type(ids) == int:
            return [self.image_triplet_dict[str(ids)]['triplets']]


def load_viroi_json(dataset_name,
                image_path,
                stuff_path,
                panoptic_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file,
                extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation annotations.

    Args:
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    timer = Timer()
    viroi_api=VIROI(image_path,stuff_path,panoptic_path,class_json_file,relation_json_file,instance_json_file,triplet_json_file)
    if timer.seconds() > 1:
        logger.info("Loading viroi takes {:.2f} seconds.".format(timer.seconds()))


    meta = MetadataCatalog.get(dataset_name)
    stuff_dataset_id_to_contiguous_id=meta.get("stuff_dataset_id_to_contiguous_id")
    thing_dataset_id_to_contiguous_id=meta.get("thing_dataset_id_to_contiguous_id")
    relation_dataset_id_to_contiguous_id=meta.get("relation_dataset_id_to_contiguous_id")
    # The categories in a custom json file may not be sorted.
    # thing_classes = [c["name"] for c in viroi_api.loadThings()]
    # meta.thing_classes = thing_classes
    # stuff_classes = [c["name"] for c in viroi_api.loadStuffs()]
    # meta.stuff_classes = stuff_classes

    # In COCO, certain category ids are artificially removed,
    # and by convention they are always ignored.
    # We deal with COCO's id issue and translate
    # the category ids to contiguous ids in [0, 80).

    # It works by looking at the "categories" field in the json, therefore
    # if users' own json also have incontiguous ids, we'll
    # apply this mapping as well but print a warning.

    # meta.thing_dataset_id_to_contiguous_id = {v['category_id']: i for i, v in enumerate(viroi_api.loadThings())} # category_id => from 0 to 79
    # meta.contiguous_id_to_thing_class_id = {i:v['class_id'] for i, v in enumerate(viroi_api.loadThings())} # from 0 to 79 => from 1 to 80
    # meta.stuff_dataset_id_to_contiguous_id = {v['category_id']: i+1 for i, v in enumerate(viroi_api.loadStuffs())} # category => from 1 to 53
    # meta.contiguous_id_to_stuff_class_id = {i+1: v['class_id'] for i, v in enumerate(viroi_api.loadStuffs())}  # from 1 to 53 => from 81 to 133

    # sort indices for reproducible results
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}

    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    logger.info("Loaded {} images in VIROI".format(len(viroi_api.image_instance_dict)))

    # ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    dataset_dict = []
    image_ids=viroi_api.loadIds()
    for image_id in image_ids:
        img_dict = viroi_api.loadImgs(image_id)[0]
        record = {}
        record["file_name"] = os.path.join(image_path, img_dict["image_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["image_id"]

        instance_dict=img_dict['instances']
        objs = []
        stfs=[]
        object_id_list=[]
        stuff_id_list=[]
        thing_count=0
        stuff_count=0
        # interest_map=np.zeros((img_dict["height"],img_dict["width"]))
        for instance_id in instance_dict:
            instance=instance_dict[instance_id]

            if viroi_api.class_dict[str(instance['class_id'])]['isthing']:
                object_id_list.append(instance_id)
                obj = {}
                obj['iscrowd']=instance['iscrowd']
                obj['labeled'] = 1 if instance['labeled'] else 0
                obj['bbox']=[instance['box'][1],instance['box'][0],instance['box'][3]-instance['box'][1],instance['box'][2]-instance['box'][0]]
                obj['category_id']=thing_dataset_id_to_contiguous_id[viroi_api.loadClassdict()[str(instance['class_id'])]['category_id']]
                obj['category_name'] = viroi_api.loadClassdict()[str(instance['class_id'])]['name']
                obj['class_id'] = instance['class_id']
                obj['segmentation']=instance['segmentation']
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                # if obj['labeled']==1:
                #     mask=mask_utils.decode(instance['segmentation'])
                #     interest_map[mask==1]=255
                objs.append(obj)
                thing_count+=1
            else:
                stuff_id_list.append(instance_id)
                stf = {}
                stf['iscrowd']=instance['iscrowd']
                stf['labeled'] = 1 if instance['labeled'] else 0
                stf['bbox']=[instance['box'][1],instance['box'][0],instance['box'][3]-instance['box'][1],instance['box'][2]-instance['box'][0]]
                stf['category_id']=stuff_dataset_id_to_contiguous_id[viroi_api.loadClassdict()[str(instance['class_id'])]['category_id']]
                stf['category_name']=viroi_api.loadClassdict()[str(instance['class_id'])]['name']
                stf['class_id'] = instance['class_id']
                stf['segmentation'] = instance['segmentation']
                stf["bbox_mode"] = BoxMode.XYWH_ABS
                # if stf['labeled']==1:
                #     mask=mask_utils.decode(instance['segmentation'])
                #     interest_map[mask==1]=1
                stfs.append(stf)
                stuff_count+=1
        record["annotations"] = objs
        # record['interest_map'] = interest_map
        # Image.fromarray(interest_map).convert('L').save("interest_map.png")
        record["stuff_annotations"] = stfs
        record["instance_ids"] = object_id_list
        record["stuff_instance_ids"] = stuff_id_list
        record["sem_seg_file_name"] = os.path.join(stuff_path, img_dict["image_name"].replace("jpg","png"))
        instance_ids=[]
        for id in object_id_list:
            instance_ids.append(id)
        for id in stuff_id_list:
            instance_ids.append(id)
        triplets = viroi_api.loadTriplets(image_id)[0]
        triplet_records=[]
        for triplet_id in triplets:
            triplet=triplets[triplet_id]
            tri={}
            tri['subject_id']=instance_ids.index(str(triplet['subject_instance_id']))
            tri['object_id']=instance_ids.index(str(triplet['object_instance_id']))
            tri['relation_id']=relation_dataset_id_to_contiguous_id[triplet['relation_id']]
            triplet_records.append(tri)
        record["triplets"] = triplet_records
        dataset_dict.append(record)
    return dataset_dict


def convert_to_coco_dict(dataset_name):
    """
    Convert a dataset in detectron2's standard format into COCO json format

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name:
            name of the source dataset
            must be registered in DatastCatalog and in detectron2's standard format
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                polygons = PolygonMasks([segmentation])
                area = polygons.area()[0].item()
            else:
                # Computing areas using bounding boxes
                bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = area
            coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)
            coco_annotation["category_id"] = reverse_id_mapper(annotation["category_id"])

            if "segmentation" in annotation:
                coco_annotation["segmentation"] = annotation["segmentation"]

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.info(f"Cached annotations in COCO format already exist: {output_file}")
        else:
            logger.info(f"Converting dataset annotations in '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            with PathManager.open(output_file, "w") as json_file:
                logger.info(f"Caching annotations in COCO format: {output_file}")
                json.dump(coco_dict, json_file)


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minitest_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get("viroi")

    dicts = load_viroi_json("coco_2017_train",
                "/media/magus/Data1/magus/Methods/data/ioid_images",
                "/media/magus/Data1/magus/Methods/data/ioid_stuff",
                "/media/magus/Data1/magus/Methods/data/ioid_panoptic",
                "/media/magus/Data1/magus/Methods/data/viroi_json/class_dict.json",
                "/media/magus/Data1/magus/Methods/data/viroi_json/relation_dict.json",
                "/media/magus/Data1/magus/Methods/data/viroi_json/train_images_dict.json",
                "/media/magus/Data1/magus/Methods/data/viroi_json/train_images_triplets_dict.json",
                extra_annotation_keys=None)
    logger.info("Done loading {} samples.".format(len(dicts)))

    # dirname = "coco-data-vis"
    # os.makedirs(dirname, exist_ok=True)
    # for d in dicts:
    #     img = np.array(Image.open(d["file_name"]))
        # visualizer = Visualizer(img, metadata=meta)
        # vis = visualizer.draw_dataset_dict(d)
        # fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        # vis.save(fpath)
