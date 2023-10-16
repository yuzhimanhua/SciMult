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

'''Classification and Link Prediction in SciDocs'''

from tqdm import tqdm
import argparse
import json
import torch
import os

from eval_helper import get_any_biencoder_component_for_infer, embed_text_psg, EXPERT_CONFIG_TO_MAP

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--expert', required=True)
parser.add_argument('--out_fn', required=True)
args = parser.parse_args()

# Specificies the device used for inference. Modifies it if necessary.
device = 'cuda:0'
dataset = args.dataset
input_data_basedir = '../data/scidocs'
output_data_basedir = f'../output'
out_fn = args.out_fn

# Reads in the model parameters
model_fn = f'../model/{args.model}'
state_dict = torch.load(model_fn)
tokenizer, biencoder = get_any_biencoder_component_for_infer(state_dict['encoder_params'])

# CLS or mean pooling
norm_rep = False
if state_dict['encoder_params']['mean_pool']:
	print('The model uses mean_pool representation, using cos distance by default')
	print('If not desirable, please fix it')
	norm_rep = True

# MoE settings
task = args.expert
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

# Loads the pretrain checkpoints.
biencoder.load_state_dict(state_dict['model_dict'])

# If using GPU for inference.
biencoder.to(device)
biencoder.eval()


# Reads in papers for evaluation.
paper_fn = f'{input_data_basedir}/{dataset}'

def paper_formatting(datum, tokenizer):
	if not datum['abstract']:
		datum['abstract'] = ''
	return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

with open(paper_fn) as fin:
	data = json.load(fin)
	paper_texts = [paper_formatting(data[paper], tokenizer) for paper in data]

with open(paper_fn) as fin:
	data = json.load(fin)
	paper_ids = [data[paper]['paper_id'] for paper in data]

print('Number of papers to eval: %d' % len(paper_texts))


# Starts embedding papers.
total_data_size = len(paper_texts)
batch_size = 64
end_idx = 0
paper_embeds = []
with torch.no_grad():
	for start_idx in tqdm(range(0, total_data_size, batch_size)):
		end_idx = start_idx + batch_size
		paper_embeds.append(embed_text_psg(paper_texts[start_idx:end_idx], tokenizer, biencoder.ctx_model, device=device, \
				     norm_rep=norm_rep, expert_id=expert_id)) 

	if end_idx < total_data_size:
		paper_embeds.append(embed_text_psg(paper_texts[end_idx:], tokenizer, biencoder.ctx_model, device=device, \
				     norm_rep=norm_rep, expert_id=expert_id)) 
		
paper_tensor = torch.cat(paper_embeds, dim=0)
print('paper tensor size: ', paper_tensor.size())

if not os.path.exists(output_data_basedir):
	os.makedirs(output_data_basedir)
with open(f'{output_data_basedir}/{out_fn}', 'w') as fout:
	for paper_id, p_emb in tqdm(zip(paper_ids, paper_tensor)):
		out = {}
		out['paper_id'] = paper_id
		out['embedding'] = [x.item() for x in p_emb.cpu()]
		fout.write(json.dumps(out)+'\n')
