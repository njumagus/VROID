# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
from typing import Any, Dict, List, Tuple, Union
import torch

from detectron2.layers import cat


class Triplets:
    """
    This class represents a list of Triplets in an image.
    It stores the attributes of Triplets (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of Triplets.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/Get a field:

       .. code-block:: python

          Triplets.gt_boxes = Boxes(...)
          print(Triplets.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in Triplets)

    2. ``len(Triplets)`` returns the number of Triplets
    3. Indexing: ``Triplets[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Triplets`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_triplets``,
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Triplets`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Triplets!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of Triplets,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Triplets of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this triplet.
        """
        return self._fields

    # Tensor-like methods
    def to(self, device: str) -> "Triplets":
        """
        Returns:
            Triplets: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Triplets(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Triplets":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Triplets` where all fields are indexed by `item`.
        """
        ret = Triplets(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return len(v)
        raise NotImplementedError("Empty Triplets does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Triplets` object is not iterable!")

    @staticmethod
    def cat(triplet_lists: List["Triplets"]) -> "Triplets":
        """
        Args:
            triplet_lists (list[Triplets])

        Returns:
            Triplets
        """
        assert all(isinstance(i, Triplets) for i in triplet_lists)
        assert len(triplet_lists) > 0
        if len(triplet_lists) == 1:
            return triplet_lists[0]

        image_size = triplet_lists[0].image_size
        for i in triplet_lists[1:]:
            assert i.image_size == image_size
        ret = Triplets(image_size)
        for k in triplet_lists[0]._fields.keys():
            values = [i.get(k) for i in triplet_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_triplets={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join(self._fields.keys()))
        return s

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_triplets={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=["
        for k, v in self._fields.items():
            s += "{} = {}, ".format(k, v)
        s += "])"
        return s
