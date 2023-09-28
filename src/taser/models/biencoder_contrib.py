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

import logging
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from dpr.data.biencoder_data import BiEncoderSample
from dpr.models.biencoder import BiEncoder, BiEncoderBatch, dot_product_scores
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState
from torch import Tensor as T
from torch import nn

logger = logging.getLogger(__name__)


def cosine_scores_with_normalization(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    q_vector = torch.nn.functional.normalize(q_vector, p=2, dim=-1)
    ctx_vectors = torch.nn.functional.normalize(ctx_vectors, p=2, dim=-1)

    return dot_product_scores(q_vector, ctx_vectors)


def onehot_max(logits):
    _, max_ind = torch.max(logits, dim=-1)
    y = torch.nn.functional.one_hot(max_ind, num_classes=logits.size(-1))
    return y


class MoEBiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        num_expert: int = 1,
        num_q_expert: int = None,
        offset_expert_id: bool = False,
    ):
        super(MoEBiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        self.num_expert = num_expert
        if num_q_expert:
            self.num_q_expert = num_q_expert
            self.num_ctx_expert = num_expert - num_q_expert
        else:
            self.num_q_expert = num_expert // 2
            self.num_ctx_expert = num_expert // 2
        logger.info("Total number of experts: %d", self.num_expert)
        logger.info("number of q experts: %d", self.num_q_expert)
        logger.info("number of ctx experts: %d", self.num_ctx_expert)
        logger.info(
            "use_infer_expert for question_model: %s", question_model.use_infer_expert
        )
        logger.info("use_infer_expert for ctx_model: %s", ctx_model.use_infer_expert)
        self.offset_expert_id = offset_expert_id

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
            if fix_encoder:
                with torch.no_grad():
                    # sequence_output, pooled_output, hidden_states = sub_model(
                    outputs = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                        input_embeds=noise_input_embeds,
                        expert_id=expert_id,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                # sequence_output, pooled_output, hidden_states = sub_model(
                outputs = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                    input_embeds=noise_input_embeds,
                    expert_id=expert_id,
                )

        # return sequence_output, pooled_output, hidden_states
        if outputs is not None:
            return outputs
        return sequence_output, pooled_output, hidden_states, None

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        q_noise_input_embeds: T = None,
        ctx_noise_input_embeds: T = None,
        encoder_type: str = None,
        representation_token_pos=0,
        task_expert_id: int = None,
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

        q_expert_ids = None
        if not self.question_model.use_infer_expert:
            q_expert_ids = torch.randint(
                low=0, high=self.num_q_expert, size=(bsz,)
            ).type(torch.int64)
            assert q_expert_ids.dtype == torch.int64

        # _q_seq, q_pooled_out, _q_hidden = self.get_representation(
        q_outputs = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            noise_input_embeds=q_noise_input_embeds,
            fix_encoder=self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
            expert_id=q_expert_ids,
        )

        # if hasattr(q_outputs, "pooled_output"):
        #     q_pooled_out = q_outputs.pooled_output
        #     q_entropy_loss = q_outputs.total_entropy_loss
        # else:
        q_pooled_out = q_outputs[1]
        q_entropy_loss = q_outputs[-1]

        if context_ids is not None:
            bsz = context_ids.shape[0]
        elif ctx_noise_input_embeds is not None:
            bsz = ctx_noise_input_embeds.shape[0]
        else:
            bsz = 1

        ctx_expert_ids = None
        if not self.ctx_model.use_infer_expert:
            ctx_expert_ids = torch.randint(
                low=self.num_q_expert,
                high=(self.num_q_expert + self.num_ctx_expert),
                size=(bsz,),
            ).type(torch.int64)
            assert ctx_expert_ids.dtype == torch.int64

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        # _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
        ctx_outputs = self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            noise_input_embeds=ctx_noise_input_embeds,
            fix_encoder=self.fix_ctx_encoder,
            representation_token_pos=representation_token_pos,
            expert_id=ctx_expert_ids,
        )

        # if hasattr(ctx_outputs, "pooled_output"):
        #     ctx_pooled_out = ctx_outputs.pooled_output
        #     ctx_entropy_loss = ctx_outputs.total_entropy_loss
        # else:
        #     ctx_pooled_out = ctx_outputs[1]
        #     ctx_entropy_loss = None
        ctx_pooled_out = ctx_outputs[1]
        ctx_entropy_loss = ctx_outputs[-1]

        entropy_loss = None
        if q_entropy_loss is not None and ctx_entropy_loss is not None:
            entropy_loss = torch.concat([q_entropy_loss, ctx_entropy_loss])

        return q_pooled_out, ctx_pooled_out, entropy_loss

    def load_state(self, saved_state: CheckpointState):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderV2(BiEncoder):
    """Like dpr.models.BiEncoder, adapted for TASER."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoderV2, self).__init__(
            question_model=question_model,
            ctx_model=ctx_model,
            fix_q_encoder=fix_q_encoder,
            fix_ctx_encoder=fix_ctx_encoder,
        )

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
            if fix_encoder:
                with torch.no_grad():
                    # sequence_output, pooled_output, hidden_states = sub_model(
                    outputs = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                # sequence_output, pooled_output, hidden_states = sub_model(
                outputs = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

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
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        # _q_seq, q_pooled_out, _q_hidden = self.get_representation(
        q_outputs = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )

        q_pooled_out = q_outputs[1]

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        # _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
        ctx_outputs = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        ctx_pooled_out = ctx_outputs[1]

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
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(
                        question, tensorizer, token_str=query_token
                    )
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(
                        tensorizer.text_to_tensor(" ".join([query_token, question]))
                    )
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


class BiEncoderNllLossV2(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
        tau: float = 0.01,
        sim_method: str = "cos",
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors, sim_method=sim_method, tau=tau)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(
        q_vector: T, ctx_vectors: T, sim_method: str = "dot", tau: float = 1.0
    ) -> T:
        f = BiEncoderNllLossV2.get_similarity_function(sim_method)
        return f(q_vector, ctx_vectors) / tau

    @staticmethod
    def get_similarity_function(sim_method: str = "cos"):
        if sim_method == "dot":
            return dot_product_scores
        elif sim_method == "cos":
            return cosine_scores_with_normalization
        raise ValueError("Unknown sim_method=%s" % sim_method)
