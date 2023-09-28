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

'''Patient-to-Article Retrieval and Patient-to-Patient Retrieval in PMC-Patients'''

from tqdm import tqdm
import argparse
import json
import torch
import os

from eval_helper import get_any_biencoder_component_for_infer, embed_text_psg, EXPERT_CONFIG_TO_MAP

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True)
parser.add_argument('--subtask', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--ckpt', required=True)
args = parser.parse_args()

# Specificies the device used for inference. Modifies it if necessary.
device = 'cuda:0'
dataset = args.dataset
subtask = args.subtask
if subtask == 'PAR':
	cand = 'article'
elif subtask == 'PPR':
	cand = 'patient'
input_data_basedir = '../data/pmc_patients'
output_data_basedir = '../output'

# Reads in the model parameters
model_fn = f'../model/{args.model}/dpr_biencoder.{args.ckpt}'
state_dict = torch.load(model_fn)
tokenizer, biencoder = get_any_biencoder_component_for_infer(state_dict['encoder_params'])

# CLS or mean pooling
norm_rep = False
if state_dict['encoder_params']['mean_pool']:
	print('The model uses mean_pool representation, using cos distance by default')
	print('If not desirable, please fix it')
	norm_rep = True

# MoE settings
task = 'multi-hop-retrieval'
use_moe = False
nexpert = None
moe_map_dict = None
if state_dict['encoder_params']['use_moe']:
	use_moe = True
	nexpert = state_dict['encoder_params']['num_expert']
	print(f'Loading a MoE model for inference with {nexpert} experts')
	moe_map_dict = EXPERT_CONFIG_TO_MAP[f'{nexpert}-expert'][task]

if moe_map_dict:
	query_expert_id = moe_map_dict['query']
	cand_expert_id = moe_map_dict['ctx']
else:
	query_expert_id = None
	cand_expert_id = None

# Loads the pretrain checkpoints.
biencoder.load_state_dict(state_dict['model_dict'])

# If using GPU for inference.
biencoder.to(device)
biencoder.eval()


# Reads in candidates.
cand_fn = f'{input_data_basedir}/{dataset}{subtask}_{cand}s_test.json'

def cand_formatting(datum, tokenizer):
	return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

with open(cand_fn) as fin:
	cands = [cand_formatting(json.loads(line), tokenizer) for line in fin]

with open(cand_fn) as fin:
	idx2cand = dict([(idx, json.loads(line)[cand]) for idx, line in enumerate(fin)])

print('Number of candidates: %d' % len(cands))

# Starts embedding candidates.
total_data_size = len(cands)
batch_size = 128
end_idx = 0
cand_embeds = []
with torch.no_grad():
	for start_idx in tqdm(range(0, total_data_size, batch_size)):
		end_idx = start_idx + batch_size
		cand_embeds.append(embed_text_psg(cands[start_idx:end_idx], tokenizer, biencoder.ctx_model, device=device, \
				    norm_rep=norm_rep, expert_id=cand_expert_id)) 

	if end_idx < total_data_size:
		cand_embeds.append(embed_text_psg(cands[end_idx:], tokenizer, biencoder.ctx_model, device=device, \
				    norm_rep=norm_rep, expert_id=cand_expert_id)) 
		
cand_tensor = torch.cat(cand_embeds, dim=0)
print('candidate tensor size: ', cand_tensor.size())


# Reads in queries for evaluation.
query_fn = f'{input_data_basedir}/{dataset}{subtask}_queries_test.json'
with open(query_fn) as fin:
	queries = [json.loads(line) for line in fin]

def query_formatting(datum, tokenizer):
	return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

print('Number of queries to eval: %d' % len(queries))


# Calculate query-candidate scores.
out = {}
cut_off = 10000
with torch.no_grad():
	for datum in tqdm(queries):
		query = datum['query']
		q_emb = embed_text_psg(query_formatting(datum, tokenizer), tokenizer, biencoder.question_model, device=device, \
				norm_rep=norm_rep, expert_id=query_expert_id)
		vals = torch.matmul(q_emb, cand_tensor.T)
		topk_indices = torch.argsort(vals.view(-1), descending=True).tolist()
		topk_indices = topk_indices[:cut_off]
		y_pred = {}
		for idx in topk_indices:
			cand = idx2cand[idx]
			score = vals[0][idx].tolist()
			y_pred[cand] = score
		out[query] = y_pred

if not os.path.exists(output_data_basedir):
	os.makedirs(output_data_basedir)
with open(f'{output_data_basedir}/{dataset}{subtask}_test_out.json', 'w') as fout:
	json.dump(out, fout)
