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

'''Link Prediction (the Reranking Setting)'''

from tqdm import tqdm
import argparse
import json
import torch
import pytrec_eval
import numpy as np

from eval_helper import get_any_biencoder_component_for_infer, embed_text_psg, EXPERT_CONFIG_TO_MAP


parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--ckpt', required=True)
args = parser.parse_args()

# Specificies the device used for inference. Modifies it if necessary.
device = 'cuda:0'
dataset = args.dataset
input_data_basedir = '../data/link_prediction_reranking'

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


# Reads in papers for evaluation.
paper_fn = f'{input_data_basedir}/{dataset}_papers_test.json'

def paper_formatting(datum, tokenizer):
	return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

with open(paper_fn) as fin:
	papers = [json.loads(line) for line in fin]

with open(paper_fn) as fin:
	paper_ids = [json.loads(line)['paper'] for line in fin]

with open(paper_fn) as fin:
	paper_texts = [paper_formatting(json.loads(line), tokenizer) for line in fin]

print('Number of papers to eval: %d' % len(papers))

# Starts embedding papers.
total_data_size = len(papers)
batch_size = 64
end_idx = 0
paper_embeds = []
with torch.no_grad():
	for start_idx in tqdm(range(0, total_data_size, batch_size)):
		end_idx = start_idx + batch_size
		# Encode twice if query_expert_id != cand_expert_id in your model.
		paper_embeds.append(embed_text_psg(paper_texts[start_idx:end_idx], tokenizer, biencoder.ctx_model, device=device, \
				     norm_rep=norm_rep, expert_id=query_expert_id))

	if end_idx < total_data_size:
		# Encode twice if query_expert_id != cand_expert_id in your model.
		paper_embeds.append(embed_text_psg(paper_texts[end_idx:], tokenizer, biencoder.ctx_model, device=device, \
				     norm_rep=norm_rep, expert_id=query_expert_id))
		
paper_tensor = torch.cat(paper_embeds, dim=0)
paper2emb = {}
for paper_id, emb in zip(paper_ids, paper_tensor):
	paper2emb[paper_id] = emb


# Calculate nDCG@k.
qrel = {}
run = {}
with torch.no_grad():
	for datum in tqdm(papers):
		paper = datum['paper']
		emb = paper2emb[paper]
		
		y = datum['score']
		if len(y) == 0:
			continue
		y_pred = {}
		for x in y:
			y_pred[x] = np.dot(emb, paper2emb[x]).item()
		qrel[paper] = y
		run[paper] = y_pred

evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg', 'ndcg_cut'})
results = evaluator.evaluate(run)

nDCG5 = pytrec_eval.compute_aggregated_measure('ndcg', [query_measures['ndcg_cut_5'] for query_measures in results.values()])
nDCG10 = pytrec_eval.compute_aggregated_measure('ndcg', [query_measures['ndcg_cut_10'] for query_measures in results.values()])
nDCG = pytrec_eval.compute_aggregated_measure('ndcg', [query_measures['ndcg'] for query_measures in results.values()])
nDCG5, nDCG10, nDCG = nDCG5*100, nDCG10*100, nDCG*100
print('nDCG@5,10,inf:', nDCG5, ',', nDCG10, ',', nDCG)
with open('scores.txt', 'a') as fout:
	fout.write('{:.2f}'.format(nDCG5)+'\t'+'{:.2f}'.format(nDCG10)+'\t'+'{:.2f}'.format(nDCG)+'\n')
