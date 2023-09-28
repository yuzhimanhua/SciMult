#!/usr/bin/env python3
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

"""Like third_party/DPR/generate_dense_embeddings.py, adapted for TASER."""

import logging
import os
import pathlib
import pickle
from typing import List, Tuple

import hydra
import numpy as np
import torch
from dpr.data.biencoder_data import BiEncoderPassage
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.cuda.amp import autocast

from taser.models import init_biencoder_components_v2
from taser.utils.model_utils_contrib import setup_for_distributed_mode_without_apex

logger = logging.getLogger()
setup_logger(logger)


def gen_ctx_vectors_v2(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
    norm_vector: bool = False,
    expert_id: int = None,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    if norm_vector:
        logger.info("Normalizing vectors for context")
    else:
        logger.info("No normalizaton for context vectors")

    for j, batch_start in enumerate(range(0, n, bsz)):
        batch = ctx_rows[batch_start : batch_start + bsz]
        batch_token_tensors = [
            tensorizer.text_to_tensor(
                ctx[1].text, title=ctx[1].title if insert_title else None
            )
            for ctx in batch
        ]

        ctx_ids_batch = move_to_device(
            torch.stack(batch_token_tensors, dim=0), cfg.device
        )
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(
            tensorizer.get_attn_mask(ctx_ids_batch), cfg.device
        )
        with torch.no_grad():
            if expert_id is None:
                outputs = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
            else:
                outputs = model(
                    ctx_ids_batch, ctx_seg_batch, ctx_attn_mask, expert_id=expert_id
                )
            out = outputs[1]
            # out = outputs.pooled_output
        out = out.cpu()
        if norm_vector:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)

        ctx_ids = [r[0] for r in batch]
        extra_info = []
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        # TODO: refactor to avoid 'if'
        if extra_info:
            results.extend(
                [
                    (ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i])
                    for i in range(out.size(0))
                ]
            )
        else:
            results.extend(
                [(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))]
            )

        if total % 10 == 0:
            logger.info("Encoded passages %d", total)
    return results


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):
    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"
    assert cfg.norm_vector is not None, "Please specify whether to normalize vector"

    cfg = setup_cfg_gpu(cfg)

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components_v2(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )

    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    encoder, _ = setup_for_distributed_mode_without_apex(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state)

    logger.info("reading data source: %s", cfg.ctx_src)

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    shard_size = len(all_passages) // cfg.num_shards
    remain_size = len(all_passages) % cfg.num_shards
    start_idx = 0
    end_idx = len(all_passages)

    if cfg.shard_id < remain_size:
        start_idx = cfg.shard_id * (shard_size + 1)
        end_idx = start_idx + shard_size + 1
    else:
        start_idx = cfg.shard_id * shard_size + remain_size
        end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    gpu_passages = all_passages[start_idx:end_idx]

    gpu_id = cfg.gpu_id
    if gpu_id == -1:
        raise ValueError("DDP inference is disabled!")

    expert_id = None
    if cfg.encoder.use_moe:
        # TODO(chenghao): Fix this.
        logger.info("Setting expert_id=1")
        expert_id = 1
    with autocast(enabled=cfg.fp16):
        data = gen_ctx_vectors_v2(
            cfg,
            gpu_passages,
            encoder,
            tensorizer,
            insert_title=cfg.encoder.use_title,
            norm_vector=cfg.norm_vector,
            expert_id=expert_id,
        )
    if gpu_id == -1:
        file = cfg.out_file + "_" + str(cfg.shard_id)
    else:
        file = cfg.out_file + "_shard" + str(cfg.shard_id) + "_gpu" + str(gpu_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)


if __name__ == "__main__":
    main()
