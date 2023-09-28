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

import collections
import itertools
import logging
import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from dpr.data.biencoder_data import BiEncoderSample
from dpr.models.biencoder import BiEncoderBatch, dot_product_scores
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState
from torch import Tensor as T
from torch import nn

logger = logging.getLogger(__name__)


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    # q_vector = _normalize(q_vector, 2, dim=1)
    # ctx_vectors = _normalize(ctx_vectors, 2, dim=1)
    q_vector = torch.nn.functional.normalize(q_vector, p=2, dim=-1)
    ctx_vectors = torch.nn.functional.normalize(ctx_vectors, p=2, dim=-1)

    return dot_product_scores(q_vector, ctx_vectors)


def gaussian_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    # q_vector = _normalize(q_vector, 2, dim=1)
    # ctx_vectors = _normalize(ctx_vectors, 2, dim=1)
    q_vector = torch.nn.functional.normalize(q_vector, p=2, dim=-1)
    ctx_vectors = torch.nn.functional.normalize(ctx_vectors, p=2, dim=-1)

    return 1.0 - dot_product_scores(q_vector, ctx_vectors)


def onehot_max(logits):
    _, max_ind = torch.max(logits, dim=-1)
    y = torch.nn.functional.one_hot(max_ind, num_classes=logits.size(-1))
    return y


def is_moe_variable(var_name, moe_layer_name="moe_layer"):
    """Checks a variable is a MoE variable or not."""
    if moe_layer_name in var_name or "moe" in var_name:
        return True
    return False


import regex
def init_moe_from_nonmoe_pretrained(pretrained_sd, moe_sd,
                                    moe_layer_name="moe_layer"):
    state_dict = {}
    missing_vars = []
    pattern_list = [
       (f"{moe_layer_name}.", ""),
       (r"interm_layers.\d+", "intermediate"),
       (r"output_layers.\d+", "output"),
       (r"moe_query.\d+", "query"),
       (r"moe_key.\d+", "key"),
       (r"moe_value.\d+", "value"),
       (r"moe_dense.\d+", "dense"),
    ]

    def normalize_var_name(var_name):
        for ptn in pattern_list:
            var_name = regex.sub(ptn[0], ptn[1], var_name)
        return var_name

    for var_name in moe_sd:
        if moe_layer_name in var_name or "moe" in var_name:
            pretrained_var_name = normalize_var_name(var_name)
            logger.info(f"Loads {var_name} from {pretrained_var_name}")
        else:
            pretrained_var_name = var_name

        if pretrained_var_name in pretrained_sd:
            state_dict[var_name] = pretrained_sd[pretrained_var_name]
        else:
            missing_vars.append((var_name, pretrained_var_name))

    again_missing_vars = []
    for var_name, _ in missing_vars:
        if "expert_gate" in var_name:
            logger.info("Random init %s", var_name)
            state_dict[var_name] = moe_sd[var_name]
        else:
            again_missing_vars.append(var_name)

    if again_missing_vars:
        print("Missing again variables:", again_missing_vars)
    return state_dict


def init_moe_from_moe_pretrained(pretrained_sd, moe_sd,
                                 from_moe_map, to_moe_map,
                                 moe_layer_name="moe_layer"):
    state_dict = {}
    missing_vars = []

    moe_index_maps = [
        (from_moe_map[key], to_moe_map[key])
        for key in to_moe_map
    ]

    print(moe_index_maps)
    pattern_list = [
       "interm_layers",
       "output_layers",
       "moe_query",
       "moe_key",
       "moe_value",
       "moe_dense",
    ]

    def normalize_var_name(var_name):
        for ptn in pattern_list:
            if ptn in var_name:
                for from_ind, to_ind in moe_index_maps:
                    from_var = f"{ptn}.{from_ind}"
                    to_var = f"{ptn}.{to_ind}"
                    if to_var in var_name:
                        var_name = regex.sub(to_var, from_var, var_name)
        return var_name

    for var_name in moe_sd:
        if moe_layer_name in var_name or "moe" in var_name:
            pretrained_var_name = normalize_var_name(var_name)
            print(f"Loads {var_name} from {pretrained_var_name}")
        else:
            pretrained_var_name = var_name

        if pretrained_var_name in pretrained_sd:
            state_dict[var_name] = pretrained_sd[pretrained_var_name]
        else:
            missing_vars.append((var_name, pretrained_var_name))

    again_missing_vars = []
    for var_name, _ in missing_vars:
        if "expert_gate" in var_name:
            logger.info("Random init %s", var_name)
            state_dict[var_name] = moe_sd[var_name]
        else:
            again_missing_vars.append(var_name)

    if again_missing_vars:
        print("Missing again variables:", again_missing_vars)
    return state_dict


def init_nonmoe_from_moe_pretrained(pretrained_moe_sd, self_sd,
                                    from_moe_map,
                                    pretrain_n_expert=None,
                                    verbose=False,
                                    moe_layer_name="moe_layer"):
    state_dict = {}
    missing_vars = []

    if from_moe_map:
        n_experts = len(from_moe_map)
        print(f"Loading from {n_experts} experts from the pretrained model")
        print(from_moe_map)
    else:
        print("Loading the average of all from the pretrained model")
        if pretrain_n_expert is None:
            raise ValueError("When from_moe_map is None, pretrained_n_expert is required!")
        n_experts = pretrain_n_expert

    pattern_list = [
       (f"{moe_layer_name}.", ""),
       (r"interm_layers.\d+", "intermediate"),
       (r"output_layers.\d+", "output"),
       (r"moe_query.\d+", "query"),
       (r"moe_key.\d+", "key"),
       (r"moe_value.\d+", "value"),
       (r"moe_dense.\d+", "dense"),
    ]

    moe_var_pattern_list = [
       "interm_layers",
       "output_layers",
       "moe_query",
       "moe_key",
       "moe_value",
       "moe_dense",
    ]

    target_var_subnames = [
        f"{moe_var_name}.{expert_idx}"
        for moe_var_name in moe_var_pattern_list
        for _, expert_idx in from_moe_map.items()
    ]

    normalize_factor = float(n_experts)

    def normalize_moe_var_name(var_name):
        for ptn in pattern_list:
            var_name = regex.sub(ptn[0], ptn[1], var_name)
        return var_name

    to_from_vars = sorted([
        (normalize_moe_var_name(from_var), from_var)
        for from_var in pretrained_moe_sd.keys()
    ])

    from_var_gps = dict([
        (to_var, [from_var for _, from_var in from_groups])
        for to_var, from_groups in itertools.groupby(
            sorted(to_from_vars, key=lambda x: x[0]), lambda x: x[0])])

    def gather_vars(weight_dict, var_list):
        tensor = weight_dict[var_list[0]]
        if len(var_list) > 1:
            for vname in var_list[1:]:
                tensor += weight_dict[vname]
        # return sum([weight_dict[vname] for vname in var_list])
        return tensor

    def is_target_from_var(var_name):
        return any([target_vname in var_name for target_vname in target_var_subnames])

    first_moe_layer = True

    for var_name in self_sd:
        # print(var_name)
        from_vars = from_var_gps.get(var_name, None)
        if from_vars:
            if len(from_vars) > 1 and from_moe_map:
                from_vars = list(filter(lambda x: is_target_from_var(x), from_vars))
            if not from_vars:
                raise ValueError("No target vars, ", from_var_gps[var_name])

            state_dict[var_name] = gather_vars(pretrained_moe_sd, from_vars)
            if len(from_vars) > 1:
                if verbose or first_moe_layer:
                    print(f"Loads {var_name} from ", ";".join(from_vars), f" w/ factor {normalize_factor}")
                    first_moe_layer = False
                state_dict[var_name] /= normalize_factor
            else:
                if verbose or "layer.0." in var_name:
                    print(f"Loads {var_name} from ", ";".join(from_vars))
        else:
            missing_vars.append((var_name, var_name))

    again_missing_vars = []
    for var_name, _ in missing_vars:
        if "expert_gate" in var_name:
            print("Random init %s", var_name)
            state_dict[var_name] = self_sd[var_name]
        else:
            again_missing_vars.append(var_name)

    if again_missing_vars:
        print("Missing again variables:", again_missing_vars)

    return state_dict


def map_span_rep_pos_v1(offset_mapping, char_start, char_end):
    """Maps the original character span positions to token ones.
    Args:
        offset_mapping: a Dict maps the token to original character start/end.
        char_start: an integer for the original span character start.
        char_end: an integer for the original span character end.
    Returns the tokenized start and end positions, both inclusive.
    """
    token_start, token_end = None, None
    if char_start is not None and char_end is not None:
        for tok_idx, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                # This is reserved for special tokens.
                continue
            if token_start is None:
                if char_start >= start and char_start <= end:
                    token_start = tok_idx
                if char_end >= start and char_end <= end:
                    token_end = tok_idx
            elif token_end is None and char_end <= end:
                token_end = tok_idx
    # if token_start is None or token_end is None:
    #     return [0, 0]
    return [token_start, token_end]


def is_overlap(seg_char_start, seg_char_end, char_start, char_end):
    """Returns either the start or end position is within the range."""
    if seg_char_start <= char_start and char_start <= seg_char_end:
        return True
    if seg_char_start <= char_end and char_end <= seg_char_end:
        return True
    return False


def get_seg_start_end(special_token_mask):
    """Finds the indices for the first and last non-special tokens."""
    start, end = 0, len(special_token_mask) - 1
    if sum(special_token_mask) == len(special_token_mask):
        raise ValueError("All tokens are special tokens!")
    while special_token_mask[start] == 1:
        start += 1
    while special_token_mask[end] == 1:
        end -= 1
    return start, end


def map_span_rep_pos(offset_mappings, special_token_masks, char_start, char_end, use_random_seg=False):
    """Maps the original character span positions to token ones."""
    token_start, token_end = None, None
    overlap_cases = []
    for offset_idx, offset_mapping in enumerate(offset_mappings):
        start_offset, end_offset = get_seg_start_end(special_token_masks[offset_idx])
        seg_start = offset_mapping[start_offset][0]
        seg_end = offset_mapping[end_offset][1]

        if seg_start > char_end:
            break

        # If the current segument does not contain the target span, skips this.
        if not is_overlap(
            seg_start, seg_end, char_start, char_end):
            continue

        # Computes the overlap length between the span and text segment.
        overlap_length = min(
            char_end - max(seg_start, char_start) + 1,
            seg_end - max(seg_start, char_start) + 1)

        overlap_cases.append((
            overlap_length,
            offset_idx,
            offset_mapping,
        ))

    # Sorts overlap cases by length.
    overlap_cases.sort(key=lambda x: -x[0])
    max_overlap_len, offset_idx, offset_mapping = overlap_cases[0]

    if use_random_seg:
        while len(overlap_cases) > 1:
            if overlap_cases[-1][0] < max_overlap_len:
                overlap_cases.pop(-1)
                continue
            break
    
        # Randomly chooses one segment if multiple segments with the max overlap length.
        if len(overlap_cases) > 1:
            random.shuffle(overlap_cases)
            max_overlap_len, offset_idx, offset_mapping = overlap_cases[0]

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            # This is reserved for special tokens.
            continue

        if token_start is None:
            if char_start >= start and char_start <= end:
                token_start = tok_idx
                token_end = tok_idx
        elif char_end >= start and char_end <= end:
            token_end = tok_idx

        if start > char_end:
            break

    return token_start, token_end, offset_idx


def get_the_target_span_segment_input(inputs, char_start, char_end, use_random_seg=False):
    """Gets the tokenized segment that contains the target span."""
    if char_start == char_end == 0:
        # Regresses to CLS token.
        tok_start, tok_end, offset_idx = 0, 0, 0
    else:
        tok_start, tok_end, offset_idx = map_span_rep_pos(
            inputs.offset_mapping,
            inputs.special_tokens_mask,
            char_start,
            char_end,
            use_random_seg=use_random_seg,
        )

    for key in inputs:
        inputs[key] = inputs[key][offset_idx]

    inputs["question_offset"] = [tok_start, tok_end]
    return inputs


def get_span_representation(seq_hiddens, span_start_end, span_method="start_end_concat"):
    """Computes the span representations using start/end positions.
    Args:
        seq_hiddens: A [bsz, seq_len, hidden_dim]-sized tensor.
        span_start_end: A [bsz, 2]-sized tensor containing boundary positions.
    Returns span representation of size [bsz, hidden_dim].
    """
    bsz, _, _ = seq_hiddens.size()
    bsz_x_span_per_sample, _ = span_start_end.size()
    ind_range = torch.arange(bsz, dtype=torch.int64)
    if bsz_x_span_per_sample != bsz:
        span_per_sample = bsz_x_span_per_sample // bsz
        ind_range = ind_range.view(-1, 1).tile((1, span_per_sample)).view(-1)
    start_embs = seq_hiddens[ind_range, span_start_end[:, 0], :]
    end_embs = seq_hiddens[ind_range, span_start_end[:, 1], :]
    if span_method == "start_end_concat":
        span_embs = torch.concat([start_embs, end_embs], dim=1)
    elif span_method == "start_end_sum":
        span_embs = (start_embs + end_embs) / 2.0
    else:
        raise ValueError("Unknown span method %s" % span_method)
    return span_embs


def entity_token_dropout(tokenized_inputs, start_pos, end_pos, mask_id, entity_drop_prob=0.3):
    if random.random() < entity_drop_prob:
        if random.random() < 0.5 and tokenized_inputs.question_offset[0] > 0:
            # Drops the entity token with masks.
            tokenized_inputs.input_ids[start_pos:end_pos + 1] = [mask_id for _ in range(start_pos, end_pos + 1)]
    return tokenized_inputs

    
def compute_neg_spans(pos_spans, start_pos, end_pos, special_tokens_mask, max_neg_spans=50, max_span_len=20):
    start_offset, end_offset = get_seg_start_end(special_tokens_mask)
    ub = min(end_offset, end_pos + max_span_len)
    lb = max(start_offset, start_pos - max_span_len)
    neg_spans = list(filter(
        lambda x: x not in pos_spans,
        [
            (ss, ee) 
            for ss in range(lb, ub)
            for ee in range(ss, ub)
            if ee - ss <= max_span_len
    ]))   
    random.shuffle(neg_spans)
    neg_spans = [(0, 0)] + neg_spans[:max_neg_spans - 1]
    return neg_spans + [(-1, -1) for _ in range(max_neg_spans - len(neg_spans))]

    
class BiEncoderSpanLoss(object):
    def calc(
        self,
        pos_scores: T,
        neg_scores: T,
        loss_scale: float = None
    ) -> Tuple[T, int]:
        span_scores = torch.concat([pos_scores, neg_scores], dim=1)
        pos_loss = F.cross_entropy(
            span_scores,
            torch.zeros(pos_scores.shape[0], dtype=torch.long).to(pos_scores.device),
            reduction="mean",
        )
        cls_loss = F.cross_entropy(
            neg_scores,
            torch.zeros(neg_scores.shape[0], dtype=torch.long).to(neg_scores.device),
            reduction="mean",
        )
        loss = (pos_loss + cls_loss) / 2
        max_score, max_idxs = torch.max(span_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.zeros(span_scores.shape[0]).to(span_scores.device)
        ).sum()

        _, max_idxs = torch.max(neg_scores, 1)
        correct_threshold_count = (
            max_idxs == torch.zeros(neg_scores.shape[0]).to(neg_scores.device)
        ).sum()

        correct_predictions_count = (correct_predictions_count + correct_threshold_count)/2
        if loss_scale:
            loss.mul_(loss_scale)
        return loss, correct_predictions_count


class MoEBiEncoderUni(nn.Module):
    """MoE Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        num_expert: int = 1,
        num_q_expert: int = None,
        q_rep_method: str = "start_end_concat",
        span_rep_method: str = "start_end_concat",
        ctx_rep_method: str = "cls",
        # entity_drop_prob: float = 0.3,
        # title_drop_prob: float = 0.1,
        use_shared_encoder: float = False,
        do_span: bool = False,
    ):
        super(MoEBiEncoderUni, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

        self.q_rep_method = q_rep_method
        # self.entity_drop_prob = entity_drop_prob
        # self.title_drop_prob = title_drop_prob
        self.ctx_rep_method = ctx_rep_method
        self.span_rep_method = span_rep_method

        self.num_expert = num_expert
        self.shared_encoder = use_shared_encoder
        self.do_span = do_span
        if use_shared_encoder:
            if num_q_expert:
                self.num_q_expert = num_q_expert
                self.num_ctx_expert = num_expert - num_q_expert
            else:
                self.num_q_expert = num_expert // 2
                self.num_ctx_expert = num_expert // 2
        else:
            self.num_q_expert = self.question_model.num_expert
            self.num_ctx_expert = self.ctx_model.num_expert

        logger.info("Total number of experts: %d", self.num_expert)
        logger.info("number of q experts: %d", self.num_q_expert)
        logger.info("number of ctx experts: %d", self.num_ctx_expert)
        logger.info("use_infer_expert for question_model: %s", question_model.use_infer_expert)
        logger.info("use_infer_expert for ctx_model: %s", ctx_model.use_infer_expert)

        #TODO(chenghao): Makes this output configurable.
        self.linear = None
        if q_rep_method == "start_end_concat" and ctx_rep_method == "cls":
            print("Using concat of start and end embeddings as representation")
            self.linear = nn.Linear(768*2, 768)
            self.linear.weight.data.normal_(mean=0.0, std=0.02)
        elif q_rep_method == "start_end_sum":
            print("Using sum of start and end embeddings as representation")
        else:
            raise NotImplementedError

        self.span_proj = None
        self.span_query = None
        if self.do_span:
            if span_rep_method == "start_end_concat":
                print("Using concat of start and end embeddings with tanh")
                hidden_dim = self.question_model.encoder.config.hidden_size
                self.span_proj = nn.Linear(hidden_dim * 2, hidden_dim)
                self.span_proj.weight.data.normal_(mean=0.0, std=0.02)
                self.span_query = nn.Linear(hidden_dim, 1)
                self.span_query.weight.data.normal_(mean=0.0, std=0.02)
            elif span_rep_method == "start_end_sum":
                print("Using sum of start and end embeddings as span representation")
            else:
                raise NotImplementedError

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        noise_input_embeds: T = None,
        fix_encoder: bool = False,
        representation_token_pos=0,
        expert_id=None,
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        outputs = None
        if ids is not None or noise_input_embeds is not None:
            inputs = {
                "input_ids": ids,
                "token_type_ids": segments,
                "attention_mask": attn_mask,
                "representation_token_pos": representation_token_pos,
            }
            if noise_input_embeds is not None:
                inputs["input_embeds"] = noise_input_embeds
            if expert_id is not None:
                inputs["expert_id"] = expert_id

            if fix_encoder:
                with torch.no_grad():
                    # outputs = sub_model(
                    #     ids,
                    #     segments,
                    #     attn_mask,
                    #     representation_token_pos=representation_token_pos,
                    # )
                    outputs = sub_model(**inputs)

                if sub_model.training:
                    for item_output in outputs:
                        item_output.requires_grad_(requires_grad=True)
                    # sequence_output.requires_grad_(requires_grad=True)
                    # pooled_output.requires_grad_(requires_grad=True)
            else:
                # sequence_output, pooled_output, hidden_states = sub_model(
                # outputs = sub_model(
                #     ids,
                #     segments,
                #     attn_mask,
                #     representation_token_pos=representation_token_pos,
                # )
                outputs = sub_model(**inputs)

        if outputs is not None:
            return outputs
        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        q_noise_input_embeds: T = None,
        ctx_noise_input_embeds: T = None,
        representation_token_pos=0,
        q_rep_token_pos: T = None,
        ctx_rep_token_pos: T = None,
        q_expert_ids: T = None,
        ctx_expert_ids: T = None,
        neg_span_pos: T = None,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )

        if question_ids is not None:
            bsz = question_ids.shape[0]
        elif q_noise_input_embeds is not None:
            bsz = q_noise_input_embeds.shape[0]
        else:
            bsz = 1

        # if self.num_expert > 1 and not self.question_model.use_infer_expert:
        # assert q_expert_ids is not None
        if q_expert_ids is None:
            raise ValueError("q_expert_id is required")
            if self.num_expert > 1 and not self.question_model.use_infer_expert:
                q_expert_ids = torch.randint(low=0, high=self.num_q_expert, size=(bsz,)).type(torch.int64)
                assert q_expert_ids.dtype == torch.int64
        else:
            assert q_expert_ids.dtype == torch.int64
            if (q_expert_ids >= self.num_expert).sum() > 0:
                raise ValueError("q_expert_ids bigger than num_expert", q_expert_ids)

        # print("query_expert_ids", q_expert_ids)
        # _q_seq, q_pooled_out, _q_hidden = self.get_representation(
        q_outputs = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            fix_encoder=self.fix_q_encoder,
            noise_input_embeds=q_noise_input_embeds,
            # representation_token_pos=q_rep_token_pos if q_rep_token_pos is not None else 0,
            expert_id=q_expert_ids,
        )

        if neg_span_pos is None:
            q_rep_method = self.q_rep_method
        else:
            q_rep_method = self.span_rep_method

        # q_pooled_out = q_outputs[1]
        if q_rep_token_pos is not None:
            # print("Using q_rep_token_pos", q_rep_token_pos)
            # print("Using q_rep_token_pos for questions")
            q_pooled_out = get_span_representation(q_outputs[0], q_rep_token_pos, span_method=q_rep_method)
        else:
            # print("Using pooled representation for questions")
            q_pooled_out = q_outputs[1]

        if self.linear is not None:
            q_pooled_out = self.linear(q_pooled_out)

        q_entropy_loss = q_outputs[-1]

        if neg_span_pos is not None:
            if not self.do_span:
                raise ValueError("When do_span=False, neg_span_pos is supposed to be None")
            n_negs = neg_span_pos.shape[1]
            flat_neg_span_pos = neg_span_pos.view(-1, 2)
            q_neg_spans = get_span_representation(q_outputs[0], flat_neg_span_pos, span_method=q_rep_method)

            # Span proposal projection.
            q_neg_proj = torch.tanh(self.span_proj(q_neg_spans))
            q_neg_scores = self.span_query(q_neg_proj)
            span_mask = (flat_neg_span_pos[:, 0] > -1).view(-1, 1).float()
            q_neg_scores += torch.log(span_mask)
            q_neg_scores = q_neg_scores.view(bsz, n_negs)

            q_pos_proj = torch.tanh(self.span_proj(q_pooled_out))
            q_pos_scores = self.span_query(q_pos_proj)

            return q_pos_scores, q_neg_scores, None

        if context_ids is not None:
            bsz = context_ids.shape[0]
        elif ctx_noise_input_embeds is not None:
            bsz = ctx_noise_input_embeds.shape[0]
        else:
            bsz = 1

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )

        # ctx_expert_ids = None
        if ctx_expert_ids is None:
            raise ValueError("ctx_expert_id is required")
            if self.num_expert > 1 and not self.ctx_model.use_infer_expert:
                if self.shared_encoder:
                    ctx_expert_ids = torch.randint(low=self.num_q_expert, high=(self.num_q_expert + self.num_ctx_expert), size=(bsz,)).type(torch.int64)
                else:
                    ctx_expert_ids = torch.randint(low=0, high=self.num_ctx_expert, size=(bsz,)).type(torch.int64)
                assert ctx_expert_ids.dtype == torch.int64
        else:
            if (ctx_expert_ids >= self.num_expert).sum() > 0:
                raise ValueError("ctx_expert_ids bigger than num_expert", ctx_expert_ids)
            # if q_expert_ids.max() >= ctx_expert_ids.min():
            #     logger.warning("query and ctx expert id overlaps!")

        # print("ctx_expert_ids", ctx_expert_ids)

        # _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
        ctx_outputs= self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            noise_input_embeds=ctx_noise_input_embeds,
            fix_encoder=self.fix_ctx_encoder,
            expert_id=ctx_expert_ids,
            # representation_token_pos=ctx_rep_token_pos if ctx_rep_token_pos is not None else 0,
        )

        ctx_pooled_out = ctx_outputs[1]
        ctx_entropy_loss = ctx_outputs[-1]

        entropy_loss = None
        if q_entropy_loss is not None and ctx_entropy_loss is not None:
            entropy_loss = torch.concat([q_entropy_loss, ctx_entropy_loss])

        if question_ids is not None and context_ids is not None:
            assert q_pooled_out.size(1) == ctx_pooled_out.size(1)
        return q_pooled_out, ctx_pooled_out, entropy_loss

    def load_state(self, saved_state: CheckpointState,
                   from_moe_map: Dict=None,
                   pretrain_n_expert: int=None,
                   to_moe_map: Dict=None):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]

        has_concat_rep_weight = "linear.weight" in saved_state.model_dict
        if has_concat_rep_weight:
            if self.q_rep_method == "start_end_concat":
                logger.info("Loading linear layer for start_end_concat q_rep_method")
            else:
                del saved_state.model_dict["linear.weight"]
                del saved_state.model_dict["linear.bias"]

        # TODO(chenghao): This is hardcoded namespace check.
        if from_moe_map:
            logger.info("Loading from a MoE checkpoint")
        else:
            logger.info("Loading from a non-MoE checkpoint")

        self_dict = self.state_dict()
        if to_moe_map:
            if from_moe_map:
                logger.info("Initialize a small MoE model from a larger one")
                logger.info(f"from {from_moe_map} to {to_moe_map}")
                updated_model_dict = init_moe_from_moe_pretrained(
                    saved_state.model_dict, self_dict, from_moe_map, to_moe_map)
            else:
                logger.info("Initializing a MoE model")
                updated_model_dict = init_moe_from_nonmoe_pretrained(
                    saved_state.model_dict, self_dict)
            self.load_state_dict(updated_model_dict)
            return
        elif from_moe_map:
            raise ValueError("Not supported yet to initialize non-MoE from MoE ckpts")
        self.load_state_dict(saved_state.model_dict)

    def get_state_dict(self):
        return self.state_dict()


class InstructBiEncoderUni(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        instruct_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        q_rep_method: str = "start_end_concat",
        ctx_rep_method: str = "cls",
        entity_drop_prob: float = 0.3,
        title_drop_prob: float = 0.1,
    ):
        super(BiEncoderUni, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.instruct_model = instruct_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

        self.q_rep_method = q_rep_method
        self.entity_drop_prob = entity_drop_prob
        self.title_drop_prob = title_drop_prob
        self.ctx_rep_method = ctx_rep_method

        #TODO(chenghao): Makes this output configurable.
        self.linear = None
        if q_rep_method == "start_end_concat" and ctx_rep_method == "cls":
            print("Using concat of start and end embeddings as representation")
            self.linear = nn.Linear(768*2, 768)
            self.linear.weight.data.normal_(mean=0.0, std=0.02)
        elif q_rep_method == "start_end_sum":
            print("Using sum of start and end embeddings as representation")
        else:
            raise NotImplementedError

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
        global_hiddens: T=None,
        global_masks: T=None,
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        outputs = None
        if ids is not None:
            inputs = {
                "input_ids": ids,
                "token_type_ids": segments,
                "attention_mask": attn_mask,
                "representation_token_pos": representation_token_pos,
            }
            if global_hiddens is not None:
                inputs["global_hiddens"] = global_hiddens
                inputs["global_masks"] = global_masks
                
            if fix_encoder:
                with torch.no_grad():
                    # outputs = sub_model(
                    #     ids,
                    #     segments,
                    #     attn_mask,
                    #     representation_token_pos=representation_token_pos,
                    # )
                    outputs = sub_model(**inputs)

                if sub_model.training:
                    for item_output in outputs:
                        item_output.requires_grad_(requires_grad=True)
            else:
                # outputs = sub_model(
                #     ids,
                #     segments,
                #     attn_mask,
                #     representation_token_pos=representation_token_pos,
                # )
                outputs = sub_model(**inputs)

        if outputs is not None:
            return outputs
        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        question_instruct_ids: T=None,
        question_instruct_attn_mask: T=None,
        context_instruct_ids: T=None,
        ctx_instruct_attn_mask: T=None,
        encoder_type: str = None,
        q_rep_token_pos: T = None,
        ctx_rep_token_pos: T = None,
        neg_span_pos: T = None,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        q_instruct_states = None
        if question_instruct_ids is not None:
            _, _, q_instruct_states = self.get_representation(
                self.instruct_model,
                question_ids,
                None,
                question_instruct_attn_mask,
                self.fix_q_encoder,
            )

        q_outputs = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            global_hiddens=q_instruct_states,
            global_masks=question_instruct_attn_mask,
            # representation_token_pos=q_rep_token_pos if q_rep_token_pos is not None else 0,
        )

        if q_rep_token_pos is not None:
            q_pooled_out = get_span_representation(q_outputs[0], q_rep_token_pos, span_method=self.q_rep_method)
        else:
            q_pooled_out = q_outputs[1]

        if self.linear is not None:
            q_pooled_out = self.linear(q_pooled_out)

        ctx_instruct_states = None
        if context_instruct_ids is not None:
            _, _, ctx_instruct_states = self.get_representation(
                self.instruct_model,
                context_ids,
                None,
                ctx_instruct_attn_mask,
                self.fix_q_encoder,
            )

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        ctx_outputs= self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            self.fix_ctx_encoder,
            global_hiddens=ctx_instruct_states,
            global_masks=ctx_instruct_attn_mask,
            # representation_token_pos=ctx_rep_token_pos if ctx_rep_token_pos is not None else 0,
        )
        ctx_pooled_out = ctx_outputs[1]

        if question_ids is not None and context_ids is not None:
            assert q_pooled_out.size(1) == ctx_pooled_out.size(1)
        return q_pooled_out, ctx_pooled_out


class BiEncoderUni(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        q_rep_method: str = "start_end_concat",
        ctx_rep_method: str = "cls",
        entity_drop_prob: float = 0.3,
        title_drop_prob: float = 0.1,
    ):
        super(BiEncoderUni, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

        self.q_rep_method = q_rep_method
        self.entity_drop_prob = entity_drop_prob
        self.title_drop_prob = title_drop_prob
        self.ctx_rep_method = ctx_rep_method

        #TODO(chenghao): Makes this output configurable.
        self.linear = None
        if q_rep_method == "start_end_concat" and ctx_rep_method == "cls":
            print("Using concat of start and end embeddings as representation")
            self.linear = nn.Linear(768*2, 768)
            self.linear.weight.data.normal_(mean=0.0, std=0.02)
        elif q_rep_method == "start_end_sum":
            print("Using sum of start and end embeddings as representation")
        else:
            raise NotImplementedError

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        outputs = None
        if ids is not None:
            inputs = {
                "input_ids": ids,
                "token_type_ids": segments,
                "attention_mask": attn_mask,
                "representation_token_pos": representation_token_pos,
            }
            if fix_encoder:
                with torch.no_grad():
                    # outputs = sub_model(
                    #     ids,
                    #     segments,
                    #     attn_mask,
                    #     representation_token_pos=representation_token_pos,
                    # )
                    outputs = sub_model(**inputs)

                if sub_model.training:
                    for item_output in outputs:
                        item_output.requires_grad_(requires_grad=True)
            else:
                # outputs = sub_model(
                #     ids,
                #     segments,
                #     attn_mask,
                #     representation_token_pos=representation_token_pos,
                # )
                outputs = sub_model(**inputs)

        if outputs is not None:
            return outputs
        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        q_rep_token_pos: T = None,
        ctx_rep_token_pos: T = None,
        neg_span_pos: T = None,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        q_outputs = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            # representation_token_pos=q_rep_token_pos if q_rep_token_pos is not None else 0,
        )

        if q_rep_token_pos is not None:
            q_pooled_out = get_span_representation(q_outputs[0], q_rep_token_pos, span_method=self.q_rep_method)
        else:
            q_pooled_out = q_outputs[1]

        if self.linear is not None:
            q_pooled_out = self.linear(q_pooled_out)

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        ctx_outputs= self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            self.fix_ctx_encoder,
            # representation_token_pos=ctx_rep_token_pos if ctx_rep_token_pos is not None else 0,
        )
        ctx_pooled_out = ctx_outputs[1]

        if question_ids is not None and context_ids is not None:
            assert q_pooled_out.size(1) == ctx_pooled_out.size(1)
        return q_pooled_out, ctx_pooled_out

    @classmethod
    def create_biencoder_input2(
        cls,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        raise ValueError("Not used anymore!")
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    ctx.text, title=ctx.title if (insert_title and ctx.title) else None
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                raise NotImplementedError
                # if query_token == "[START_ENT]":
                #     query_span = _select_span_with_token(
                #         question, tensorizer, token_str=query_token
                #     )
                #     question_tensors.append(query_span)
                # else:
                #     question_tensors.append(
                #         tensorizer.text_to_tensor(" ".join([query_token, question]))
                #     )
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    @classmethod
    def create_biencoder_uni_input(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
        title_drop_prob: float = 0.0,
        entity_drop_prob: float = 0.0,
        flip_route_prob: float = 0.0,
        drop_to_retrieval_prob: float = 0.0,
        span_proposal_prob: float = 0.0,
        data_type_to_expert_map: Dict = None,
        max_neg_spans: int = 200,
        use_random_seg: bool = False,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        current_ctxs_len = 0

        sampled_ctx_input1 = []
        question_repo_pos = []
        question_attn_mask_list = []

        query_expert_ids, ctx_expert_ids = [], []
        flip_query_expert_ids, flip_ctx_expert_ids = [], []
        data_type = None

        pid_to_inbatch_index = collections.defaultdict(list)
        positive_pids = []
        max_n_pos = 1

        is_span_proposal_case = False
        neg_span_pos = []
        if random.random() < span_proposal_prob:
            is_span_proposal_case = True

        fwd_as_retrieval = False
        if random.random() < drop_to_retrieval_prob:
            fwd_as_retrieval = True

        def process_single_ctx(ctx):
            if insert_title and ctx.title and random.random() > title_drop_prob:
                # TODO(chenghao): This can be problematic for models wo/ [SEP] token.
                return " ".join([
                    ctx.title, tensorizer.sep_token, ctx.text])
            return ctx.text

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            if len(sample.positive_passages) > max_n_pos:
                max_n_pos = len(sample.positive_passages)

            # Adds the positive passage id.
            # positive_pids.append([pctx.psg_id for pctx in sample.positive_passages])

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages

            if hasattr(sample, "hop1_passages") and sample.hop1_passages:
                hop1_ctx = process_single_ctx(sample.hop1_passages[0])
                question = " ".join([
                    sample.query, tensorizer.sep_token, hop1_ctx])
            elif len(sample.query.split("\t")) > 1:
                spaced_sep_tk = " {sep_tok} ".format(sep_tok=tensorizer.sep_token)

                # If query contains tab, then concats them with sep_token.
                if random.random() > title_drop_prob:
                    question = spaced_sep_tk.join(sample.query.split("\t"))
                else:
                    question = spaced_sep_tk.join(sample.query.split("\t")[1:])
            else:
                question = sample.query

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            # Always uses hard negative if available.
            # neg_ctxs = neg_ctxs[0:num_other_negatives]
            # hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]
            all_neg_ctxs = hard_neg_ctxs + neg_ctxs
            neg_ctxs_to_use = all_neg_ctxs[0:num_hard_negatives]

            # all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            all_ctxs = [positive_ctx] + neg_ctxs_to_use
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(neg_ctxs_to_use)

            if hasattr(sample, "data_type") and sample.data_type:
                if data_type is None:
                    data_type = sample.data_type

                if is_span_proposal_case and sample.data_type == "entity-retrieval":
                    data_type = "entity-span"
                elif fwd_as_retrieval:
                    data_type = "single-hop-retrieval"
                elif sample.data_type != data_type:
                    raise ValueError("Multiple data type per batch is not supported!")

                if (flip_route_prob > 0 and random.random() < flip_route_prob
                        and "single-hop-retrieval" in data_type):
                    flip_query_expert_ids.append(data_type_to_expert_map[data_type]["query"])
                    query_expert_ids.append(data_type_to_expert_map[data_type]["ctx"])
                    flip_ctx_expert_ids.extend([
                        data_type_to_expert_map[data_type]["ctx"] for _ in range(len(all_ctxs))])
                    ctx_expert_ids.extend([
                        data_type_to_expert_map[data_type]["query"] for _ in range(len(all_ctxs))])
                else:
                    query_expert_ids.append(data_type_to_expert_map[data_type]["query"])
                    flip_query_expert_ids.append(data_type_to_expert_map[data_type]["ctx"])
                    ctx_expert_ids.extend([
                        data_type_to_expert_map[data_type]["ctx"] for _ in range(len(all_ctxs))])
                    flip_ctx_expert_ids.extend([
                        data_type_to_expert_map[data_type]["query"] for _ in range(len(all_ctxs))])

            for ii, ctx in enumerate(all_ctxs):
                sampled_ctx_input1.append(process_single_ctx(ctx))
                # pid_to_inbatch_index[ctx.psg_id].append(current_ctxs_len + ii)

            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            # Increments the number of sampled context.
            # current_ctxs_len += (1 + len(neg_ctxs_to_use))
            current_ctxs_len += len(all_ctxs)

            if data_type is not None and "entity" in data_type:
                return_overflowing_tokens = True
            else:
                return_overflowing_tokens = False
                is_span_proposal_case = False

            orig_encoded_question_inputs = tensorizer.encode_text(
                question,
                return_overflowing_tokens=return_overflowing_tokens,
                return_tensors=None,
            )

            neg_spans = None
            if return_overflowing_tokens:
                all_encoded_question_inputs = get_the_target_span_segment_input(
                    orig_encoded_question_inputs,
                    sample.query_rep_start,
                    sample.query_rep_end,
                    use_random_seg=use_random_seg,
                )
                question_offset = all_encoded_question_inputs.question_offset
                if question_offset[0] is None or question_offset[1] is None:
                    logger.warning("Can not find start or end for")
                    logger.warning(sample)
                    raise ValueError("Fail to get the span positions.")
                entity_token_dropout(
                    all_encoded_question_inputs,
                    question_offset[0],
                    question_offset[1],
                    mask_id=tensorizer.mask_token_id,
                    entity_drop_prob=entity_drop_prob,
                )

                # if question_offset [0] != all_encoded_question_inputs.question_offset[0]:
                #     logger.info("Entity is dropped")
                question_offset = all_encoded_question_inputs.question_offset

                if is_span_proposal_case:
                    neg_spans = compute_neg_spans(
                        (question_offset[0], question_offset[1]),
                        question_offset[0],
                        question_offset[1],
                        all_encoded_question_inputs.special_tokens_mask,
                        max_neg_spans=max_neg_spans,
                        )
                    assert len(neg_spans) == max_neg_spans
            else:
                all_encoded_question_inputs = orig_encoded_question_inputs
                question_offset = [0, 0]

            # print(tensorizer.tokenizer.decode(all_encoded_question_inputs.input_ids))
            q_tensor = torch.tensor(all_encoded_question_inputs.input_ids)
            q_attn_mask = torch.tensor(all_encoded_question_inputs.attention_mask)

            question_tensors.append(q_tensor)
            question_attn_mask_list.append(q_attn_mask)
            question_repo_pos.append(torch.tensor(question_offset))
            if neg_spans:
                neg_span_pos.append(torch.tensor(neg_spans))

        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)
        questions_attn_mask = torch.cat([q.view(1, -1) for q in question_attn_mask_list], dim=0)
        questions_rep_pos = torch.cat([q.view(1, -1) for q in question_repo_pos], dim=0)
        # question_segments = torch.zeros_like(questions_tensor)
        question_segments = None

        if is_span_proposal_case:
            questions_neg_pos = torch.cat([q.unsqueeze(0) for q in neg_span_pos], dim=0)

        all_encoded_ctx_inputs = tensorizer.encode_text(sampled_ctx_input1, None)
        ctxs_tensor = all_encoded_ctx_inputs.input_ids
        if hasattr(all_encoded_ctx_inputs, "token_type_ids"):
            ctx_segments = all_encoded_ctx_inputs.token_type_ids
        else:
            ctx_segments = None
        ctx_attn_mask = all_encoded_ctx_inputs.attention_mask

        # Skips all in batch samples as negative if colliding with positives.
        # donot_use_tensors = []
        # for pids in positive_pids:
        #     donot_use_indices = list(set(sum([pid_to_inbatch_index[pid] for pid in pids], [])))
        #     donot_use_indices = donot_use_indices + [-100 for _ in range(max_n_pos - len(donot_use_indices))]
        #     donot_use_tensors.append(torch.LongTensor(donot_use_indices))

        return BiEncoderBatch(
            question_ids=questions_tensor,
            question_segments=question_segments,
            question_attn_mask=questions_attn_mask,
            question_rep_pos=questions_rep_pos,
            context_ids=ctxs_tensor,
            ctx_segments=ctx_segments,
            ctx_attn_mask=ctx_attn_mask,
            ctx_rep_pos=None,
            is_positive=positive_ctx_indices,
            hard_negatives=hard_neg_ctx_indices,
            query_expert_ids=torch.LongTensor(query_expert_ids) if query_expert_ids else None,
            ctx_expert_ids=torch.LongTensor(ctx_expert_ids) if ctx_expert_ids else None,
            flip_query_expert_ids=torch.LongTensor(flip_query_expert_ids) if query_expert_ids else None,
            flip_ctx_expert_ids=torch.LongTensor(flip_ctx_expert_ids) if ctx_expert_ids else None,
            neg_span_pos=questions_neg_pos if is_span_proposal_case else None,
            # donot_use_as_negative=torch.cat(donot_use_tensors),

            #TODO(chenghao): To add those variables.
            pid_tensors=None,
            full_positive_pids=None,
            encoder_type="question",
        )

    def load_state(self, saved_state: CheckpointState,
                   from_moe_map: Dict=None,
                   pretrain_n_expert: int=None,
                   to_moe_map: Dict=None):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        has_concat_rep_weight = "linear.weight" in saved_state.model_dict
        if has_concat_rep_weight:
            if self.q_rep_method == "start_end_concat":
                logger.info("Loading linear layer for start_end_concat q_rep_method")
            else:
                del saved_state.model_dict["linear.weight"]
                del saved_state.model_dict["linear.bias"]

        if from_moe_map:
            logger.info("Loading from a MoE checkpoint")
            updated_state_dict = init_nonmoe_from_moe_pretrained(
                saved_state.model_dict,
                self.get_state_dict(),
                from_moe_map,
                pretrain_n_expert=pretrain_n_expert,
            )
            self.load_state_dict(updated_state_dict)
            return

        logger.info("Loading from a non-MoE checkpoint")
        self.load_state_dict(saved_state.model_dict)

    def get_state_dict(self):
        return self.state_dict()
