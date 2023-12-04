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

'''Evaluation Helper Functions'''

import torch
from taser.models.hf_models_contrib import HFEncoder, get_any_tokenizer
from taser.models.biencoder_contrib import BiEncoderV2, MoEBiEncoder
from taser.models.biencoder_uni_contrib import BiEncoderUni, MoEBiEncoderUni
from dpr.utils.model_utils import move_to_device

EXPERT_CONFIG_TO_MAP = {
    '3-expert': {
        'single-hop-retrieval': {
            'query': 0,
            'ctx': 0,
        },
        'multi-hop-retrieval': {
            'query': 1,
            'ctx': 1,
        },
        'entity-retrieval': {
            'query': 2,
            'ctx': 2,
        },
    },
}

def get_any_biencoder_component_for_infer(cfg, use_uni_biencoder: bool = True, **kwargs):
	'''Loads the proper tokenizer and the biencoder based on pretrained model config.
	Args:
		cfg: Dict contains necessary model configuration parameters.
		use_uni_biencoder: Bool indicates whether uniencoder model is used.
		
	Returns the corresponding tokenizer and biencoder model.
	'''
	dropout = 0.0
	use_vat = False
	use_moe = cfg['use_moe']
	num_expert = cfg['num_expert'] if use_moe else 0
	use_infer_expert = cfg['use_infer_expert'] if use_moe else False
	per_layer_gating = cfg['per_layer_gating'] if use_moe else False
	moe_type = cfg['moe_type'] if use_moe else None
	num_q_expert = 1
	num_ctx_expert = 1

	mean_pool_q_encoder = False
	if cfg['mean_pool']:
		mean_pool_q_encoder = True

	factor_rep = cfg['factor_rep']

	if cfg["pretrained_model_cfg"] == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract':
		cfg["pretrained_model_cfg"] = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

	question_encoder = HFEncoder(
		cfg['pretrained_model_cfg'],
		projection_dim=cfg['projection_dim'],
		dropout=dropout,
		use_vat=use_vat,
		use_moe=use_moe,
		moe_type=moe_type,
		use_infer_expert=use_infer_expert,
		per_layer_gating=per_layer_gating,
		num_expert=num_expert if cfg['shared_encoder'] else num_q_expert,
		mean_pool=mean_pool_q_encoder,
		factor_rep=factor_rep,
		pretrained=True,
		use_norm_rep=False,
		task_header_type=None,
		**kwargs
	)

	if cfg['shared_encoder']:
		print('Uses a shared encoder for both question and context.')
		ctx_encoder = question_encoder
	else:
		ctx_encoder = HFEncoder(
			cfg['pretrained_model_cfg'],
			projection_dim=cfg['projection_dim'],
			use_vat=use_vat,
			use_moe=use_moe,
			moe_type=moe_type,
			use_infer_expert=use_infer_expert,
			per_layer_gating=per_layer_gating,
			num_expert=num_ctx_expert if num_ctx_expert else num_expert,
			dropout=dropout,
			mean_pool=cfg['mean_pool'],
			factor_rep=factor_rep,
			pretrained=True,
			use_norm_rep=False,
			task_header_type=None,
			**kwargs
		)

	fix_ctx_encoder = False
	if use_uni_biencoder:
		if use_moe or use_vat:
			biencoder = MoEBiEncoderUni(
				question_encoder, ctx_encoder,
				fix_ctx_encoder=fix_ctx_encoder,
				num_expert=num_expert if use_moe else 0,
				num_q_expert=num_q_expert if use_moe else None,
				q_rep_method=cfg['q_rep_method'],
				do_span=False,
			)
		else:
			biencoder = BiEncoderUni(
				question_encoder, ctx_encoder,
				fix_ctx_encoder=fix_ctx_encoder,
				q_rep_method=cfg['q_rep_method'],
			)
	elif use_moe:
		print('Using MOE model')
		if num_q_expert is not None:
			offset_expert_id = True
		else:
			offset_expert_id = False
		biencoder = MoEBiEncoder(
			question_encoder,
			ctx_encoder,
			fix_ctx_encoder=fix_ctx_encoder,
			num_expert=num_expert,
			num_q_expert=num_q_expert,
			offset_expert_id=offset_expert_id,
		)
	else:
		biencoder = BiEncoderV2(
			question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)
		
	tensorizer = get_any_tokenizer(cfg['pretrained_model_cfg'])

	return tensorizer, biencoder


def embed_text_psg(text_psg, tokenizer, embed_model, norm_rep=False, max_len=512, expert_id=None, device='cuda:0'):
	'''Generates text embeddings.'''
	inputs = tokenizer(
		text_psg,
		add_special_tokens=True,
		max_length=max_len,
		padding='max_length',
		truncation=True,
		return_offsets_mapping=False,
		return_tensors='pt',
	)
	model_inputs = {
		'input_ids': move_to_device(inputs.input_ids, device=device),
		'token_type_ids': None,
		'attention_mask': move_to_device(inputs.attention_mask, device=device),
	}

	if expert_id is not None:
		model_inputs['expert_id'] = expert_id

	outputs = embed_model(**model_inputs)
	if norm_rep:
		return torch.nn.functional.normalize(outputs[1], p=2, dim=-1).cpu()
	return outputs[1].cpu()
