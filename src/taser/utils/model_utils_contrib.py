# Copyright (c) 2023 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Tuple

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def setup_for_distributed_mode_without_apex(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: object,
    n_gpu: int = 1,
    local_rank: int = -1,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Like dpr.utils.model_utils.setup_for_distributed_mode, except removing apex support."""
    model.to(device)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device if device else local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            # TODO(chenghao): fix out what does this mean?
            broadcast_buffers=False,
        )
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer


# TODO(hafang): Uncomment this if original move_to_device does not work.
# def move_to_device_wrapper(sample, device):
#     from dpr.utils.model_utils import move_to_device
#     if sample is None:
#         return None
#
#     return move_to_device(sample, device)


def get_schedule_linear_without_warmup(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):
    """Like dpr.utils.model_utils.get_linear_schedule, but without warmp up."""

    def nowarmup_lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(1.0)
        return max(
            1e-6,
            float(total_training_steps - current_step)
            / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, nowarmup_lr_lambda, last_epoch)
