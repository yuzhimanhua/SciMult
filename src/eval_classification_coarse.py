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

'''Coarse-Grained Paper Classification'''

from tqdm import tqdm
import argparse
import json
import torch
from sklearn.metrics import f1_score

from eval_helper import get_any_biencoder_component_for_infer, embed_text_psg, EXPERT_CONFIG_TO_MAP


parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()

# Specificies the device used for inference. Modifies it if necessary.
device = 'cuda:0'
dataset = args.dataset
input_data_basedir = '../data/classification_coarse'

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
task = 'entity-retrieval'
use_moe = False
nexpert = None
moe_map_dict = None
if state_dict['encoder_params']['use_moe']:
	use_moe = True
	nexpert = state_dict['encoder_params']['num_expert']
	print(f'Loading a MoE model for inference with {nexpert} experts')
	moe_map_dict = EXPERT_CONFIG_TO_MAP[f'{nexpert}-expert'][task]

if moe_map_dict:
	paper_expert_id = moe_map_dict['query']
	label_expert_id = moe_map_dict['ctx']
else:
	paper_expert_id = None
	label_expert_id = None

# Loads the pretrain checkpoints.
biencoder.load_state_dict(state_dict['model_dict'])

# If using GPU for inference.
biencoder.to(device)
biencoder.eval()


# Reads in labels.
label_fn = f'{input_data_basedir}/{dataset}_label.json'

def label_formatting(datum, tokenizer):
	return ' '.join([datum['name'][0], tokenizer.sep_token, datum['definition']])

with open(label_fn) as fin:
	labels = [label_formatting(json.loads(line), tokenizer) for line in fin]

with open(label_fn) as fin:
	idx2label = dict([(idx, json.loads(line)['label']) for idx, line in enumerate(fin)])

print('Number of labels: %d' % len(labels))

# Starts embedding labels.
total_data_size = len(labels)
batch_size = 64
end_idx = 0
label_embeds = []
with torch.no_grad():
	for start_idx in tqdm(range(0, total_data_size, batch_size)):
		end_idx = start_idx + batch_size
		label_embeds.append(embed_text_psg(labels[start_idx:end_idx], tokenizer, biencoder.ctx_model, device=device, \
				     norm_rep=norm_rep, expert_id=label_expert_id)) 

	if end_idx < total_data_size:
		label_embeds.append(embed_text_psg(labels[end_idx:], tokenizer, biencoder.ctx_model, device=device, \
				     norm_rep=norm_rep, expert_id=label_expert_id)) 
		
label_tensor = torch.cat(label_embeds, dim=0)
print('label tensor size: ', label_tensor.size())


# Reads in papers for evaluation.
paper_fn = f'{input_data_basedir}/{dataset}_papers_test.json'
with open(paper_fn) as fin:
	papers = [json.loads(line) for line in fin]

def paper_formatting(datum, tokenizer):
	return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

print('Number of papers to eval: %d' % len(papers))


# Calculate Micro/Macro-F1.
y = []
y_pred = []
with torch.no_grad():
	for datum in tqdm(papers):
		p_emb = embed_text_psg(paper_formatting(datum, tokenizer), tokenizer, biencoder.question_model, device=device, \
			    norm_rep=norm_rep, expert_id=paper_expert_id)
		y.append(datum['label'])
		vals = torch.matmul(p_emb, label_tensor.T)
		y_pred.append(torch.argsort(vals.view(-1), descending=True).tolist()[0])

MiF1 = f1_score(y, y_pred, average='micro')*100
MaF1 = f1_score(y, y_pred, average='macro')*100
print('Micro-F1 and Macro-F1:', MiF1, ',', MaF1)
with open('scores.txt', 'a') as fout:
	fout.write('{:.2f}'.format(MiF1)+'\t'+'{:.2f}'.format(MaF1)+'\n')
