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

'''Link Prediction (the Retrieval Setting)'''

from tqdm import tqdm
import argparse
import json
import torch

from eval_helper import get_any_biencoder_component_for_infer, embed_text_psg, EXPERT_CONFIG_TO_MAP


parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()

# Specificies the device used for inference. Modifies it if necessary.
device = 'cuda:0'
dataset = args.dataset
input_data_basedir = '../data/link_prediction_retrieval'

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


if dataset == 'SciDocs':
	# Reads in papers for evaluation.
	paper_fn = f'{input_data_basedir}/{dataset}_papers_test.json'

	def paper_formatting(datum, tokenizer):
		return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

	with open(paper_fn) as fin:
		papers = [json.loads(line) for line in fin]

	with open(paper_fn) as fin:
		paper_texts = [paper_formatting(json.loads(line), tokenizer) for line in fin]

	with open(paper_fn) as fin:
		idx2paper = dict([(idx, json.loads(line)['paper']) for idx, line in enumerate(fin)])

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
	print('paper tensor size: ', paper_tensor.size())


	# Calculate Recall@k.
	r20_cite = r50_cite = r100_cite = cnt_cite = 0.0
	r20_cocite = r50_cocite = r100_cocite = cnt_cocite = 0.0
	with torch.no_grad():
		for datum, p_emb in tqdm(zip(papers, paper_tensor)):
			if len(datum['cite']) == 0 and len(datum['co-cite']) == 0:
				continue
			vals = torch.matmul(p_emb, paper_tensor.T)
			topk_indices = torch.argsort(vals.view(-1), descending=True).tolist()

			# Cite
			y = datum['cite']
			if len(y) != 0:
				y_pred = [idx2paper[x] for x in topk_indices]
				tp = [x for x in y_pred[:20] if x in y]
				r20_cite += len(tp)/len(y)
				tp = [x for x in y_pred[:50] if x in y]
				r50_cite += len(tp)/len(y)
				tp = [x for x in y_pred[:100] if x in y]
				r100_cite += len(tp)/len(y)
				cnt_cite += 1

			# Co-cite
			y = datum['co-cite']
			if len(y) != 0:
				y_pred = [idx2paper[x] for x in topk_indices]
				tp = [x for x in y_pred[:20] if x in y]
				r20_cocite += len(tp)/len(y)
				tp = [x for x in y_pred[:50] if x in y]
				r50_cocite += len(tp)/len(y)
				tp = [x for x in y_pred[:100] if x in y]
				r100_cocite += len(tp)/len(y)
				cnt_cocite += 1

	r20, r50, r100 = r20_cite/cnt_cite*100, r50_cite/cnt_cite*100, r100_cite/cnt_cite*100
	print('Recall@20,50,100 (cite):', r20, ',', r50, ',', r100)
	with open('scores.txt', 'a') as fout:
		fout.write('{:.2f}'.format(r20)+'\t'+'{:.2f}'.format(r50)+'\t'+'{:.2f}'.format(r100)+'\n')

	r20, r50, r100 = r20_cocite/cnt_cocite*100, r50_cocite/cnt_cocite*100, r100_cocite/cnt_cocite*100
	print('Recall@20,50,100 (co-cite):', r20, ',', r50, ',', r100)
	with open('scores.txt', 'a') as fout:
		fout.write('{:.2f}'.format(r20)+'\t'+'{:.2f}'.format(r50)+'\t'+'{:.2f}'.format(r100)+'\n')


elif dataset == 'PMCPatientsPPR':
	# Reads in patients.
	patient_fn = f'{input_data_basedir}/{dataset}_patients_test.json'

	def patient_formatting(datum, tokenizer):
		return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

	with open(patient_fn) as fin:
		patients = [patient_formatting(json.loads(line), tokenizer) for line in fin]

	with open(patient_fn) as fin:
		idx2patient = dict([(idx, json.loads(line)['patient']) for idx, line in enumerate(fin)])

	print('Number of patients: %d' % len(patients))

	# Starts embedding patients.
	total_data_size = len(patients)
	batch_size = 128
	end_idx = 0
	patient_embeds = []
	with torch.no_grad():
		for start_idx in tqdm(range(0, total_data_size, batch_size)):
			end_idx = start_idx + batch_size
			patient_embeds.append(embed_text_psg(patients[start_idx:end_idx], tokenizer, biencoder.ctx_model, device=device, \
						   norm_rep=norm_rep, expert_id=cand_expert_id)) 

		if end_idx < total_data_size:
			patient_embeds.append(embed_text_psg(patients[end_idx:], tokenizer, biencoder.ctx_model, device=device, \
						   norm_rep=norm_rep, expert_id=cand_expert_id)) 
			
	patient_tensor = torch.cat(patient_embeds, dim=0)
	print('patient tensor size: ', patient_tensor.size())


	# Reads in queries for evaluation.
	query_fn = f'{input_data_basedir}/{dataset}_queries_test.json'
	with open(query_fn) as fin:
		queries = [json.loads(line) for line in fin]

	def query_formatting(datum, tokenizer):
		return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

	print('Number of queries to eval: %d' % len(queries))


	# Calculate Recall@k.
	r20 = r50 = r100 = cnt = 0.0
	with torch.no_grad():
		for datum in tqdm(queries):
			q_emb = embed_text_psg(query_formatting(datum, tokenizer), tokenizer, biencoder.question_model, device=device, \
					norm_rep=norm_rep, expert_id=query_expert_id)
			y = datum['patient']
			vals =  torch.matmul(q_emb, patient_tensor.T)
			topk_indices = torch.argsort(vals.view(-1), descending=True).tolist()

			y_pred = [idx2patient[x] for x in topk_indices]
			tp = [x for x in y_pred[:20] if x in y]
			r20 += len(tp)/len(y)
			tp = [x for x in y_pred[:50] if x in y]
			r50 += len(tp)/len(y)
			tp = [x for x in y_pred[:100] if x in y]
			r100 += len(tp)/len(y)
			cnt += 1

	r20, r50, r100 = r20/cnt*100, r50/cnt*100, r100/cnt*100
	print('Recall@20,50,100:', r20, ',', r50, ',', r100)
	with open('scores.txt', 'a') as fout:
		fout.write('{:.2f}'.format(r20)+'\t'+'{:.2f}'.format(r50)+'\t'+'{:.2f}'.format(r100)+'\n')
