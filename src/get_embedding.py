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

'''Sample Code to Get Text Embedding'''

import torch
from eval_helper import get_any_biencoder_component_for_infer, embed_text_psg, EXPERT_CONFIG_TO_MAP

device = 'cuda:0'

# Specify the model (vanilla/moe).
model_fn = f'../model/scimult_moe.ckpt'
state_dict = torch.load(model_fn)
tokenizer, biencoder = get_any_biencoder_component_for_infer(state_dict['encoder_params'])

norm_rep = False
if state_dict['encoder_params']['mean_pool']:
	print('The model uses mean_pool representation, using cos distance by default')
	print('If not desirable, please fix it')
	norm_rep = True

# Specify the expert.
task = 'entity-retrieval'  # If you want to use the classification expert
# task = 'multi-hop-retrieval'  # If you want to use the link prediction expert
# task = 'single-hop-retrieval'  # If you want to use the search expert

use_moe = False
nexpert = None
moe_map_dict = None
if state_dict['encoder_params']['use_moe']:
	use_moe = True
	nexpert = state_dict['encoder_params']['num_expert']
	print(f'Loading a MoE model for inference with {nexpert} experts')
	moe_map_dict = EXPERT_CONFIG_TO_MAP[f'{nexpert}-expert'][task]

if moe_map_dict:
	expert_id = moe_map_dict['query']
else:
	expert_id = None

del state_dict['model_dict']['question_model.encoder.embeddings.position_ids']
del state_dict['model_dict']['ctx_model.encoder.embeddings.position_ids']
biencoder.load_state_dict(state_dict['model_dict'])

# If using GPU for inference.
biencoder.to(device)
biencoder.eval()

# Sample input with 2 papers.
paper_texts = ['BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', \
			   'Language Models are Few-Shot Learners']

# Embedding papers.
total_data_size = len(paper_texts)
batch_size = 64
end_idx = 0
paper_embeds = []
with torch.no_grad():
	for start_idx in range(0, total_data_size, batch_size):
		end_idx = start_idx + batch_size
		paper_embeds.append(embed_text_psg(paper_texts[start_idx:end_idx], tokenizer, biencoder.ctx_model, device=device, \
					 norm_rep=norm_rep, expert_id=expert_id)) 

	if end_idx < total_data_size:
		paper_embeds.append(embed_text_psg(paper_texts[end_idx:], tokenizer, biencoder.ctx_model, device=device, \
					 norm_rep=norm_rep, expert_id=expert_id)) 
paper_tensor = torch.cat(paper_embeds, dim=0)

# Print paper embeddings.
for p_emb in paper_tensor:
	print([x.item() for x in p_emb.cpu()])
