# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
from typing import Any, Iterable, List, Tuple, Type

import torch
from torch import nn


BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


# pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because its
#  type `no_grad` is not callable.
@torch.no_grad()
def update_bn_stats(
    model: nn.Module, data_loader: Iterable[Any], num_iters: int = 200  # pyre-ignore
) -> None:
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.

    Args:
        model (nn.Module): the model whose bn stats will be recomputed.

            Note that:

            1. This function will not alter the training mode of the given model.
               Users are responsible for setting the layers that needs
               precise-BN to training mode, prior to calling this function.

            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations.  This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers, or backup the states.
        data_loader (iterator): an iterator. Produce data as inputs to the model.
        num_iters (int): number of iterations to compute the stats.
    """
    bn_layers = get_bn_modules(model)

    if len(bn_layers) == 0:
        return

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]  # pyre-ignore
    for bn in bn_layers:
        bn.momentum = 1.0

    # Note that PyTorch's running_var means "running average of
    # bessel-corrected batch variance". (PyTorch's BN normalizes by biased
    # variance, but updates EMA by unbiased (bessel-corrected) variance).
    # So we estimate population variance by "simple average of bessel-corrected
    # batch variance". This is the same as in the BatchNorm paper, Sec 3.1.
    # This estimator converges to population variance as long as batch size
    # is not too small, and total #samples for PreciseBN is large enough.
    # Its convergence may be affected by small batch size.

    # Alternatively, one can estimate population variance by the sample variance
    # of all batches combined. However, this needs a way to know the batch size
    # of each batch in this function (otherwise we only have access to the
    # bessel-corrected batch variance given by pytorch), which is an extra
    # requirement.
    running_mean = [
        torch.zeros_like(bn.running_mean) for bn in bn_layers  # pyre-ignore
    ]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]  # pyre-ignore

    ind = -1
    for ind, inputs in enumerate(itertools.islice(data_loader, num_iters)):
        model(inputs)

        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
            running_var[i] += (bn.running_var - running_var[i]) / (ind + 1)
    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind)
    )

    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]


def get_bn_modules(model: nn.Module) -> List[nn.Module]:
    """
    Find all BatchNorm (BN) modules that are in training mode. See
    fvcore.precise_bn.BN_MODULE_TYPES for a list of all modules that are
    included in this search.

    Args:
        model (nn.Module): a model possibly containing BN modules.

    Returns:
        list[nn.Module]: all BN modules in the model.
    """
    # Finds all the bn layers.
    bn_layers = [
        m for m in model.modules() if m.training and isinstance(m, BN_MODULE_TYPES)
    ]
    return bn_layers
