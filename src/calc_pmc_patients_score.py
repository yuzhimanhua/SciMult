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

'''Calculate the leaderboard metrics for PMC-Patients output.'''

import json
import argparse
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True)
parser.add_argument('--subtask', required=True)
args = parser.parse_args()

dataset = args.dataset
subtask = args.subtask
input_data_basedir = f'../data/pmc_patients/{dataset}'

corpus_path = f'{input_data_basedir}/{subtask}/{subtask}_corpus.jsonl'
query_path = f'{input_data_basedir}/queries/test_queries.jsonl'
qrels_path = f'{input_data_basedir}/{subtask}/{subtask}_test_qrels.tsv'
corpus, queries, qrels = GenericDataLoader(corpus_file=corpus_path, query_file=query_path, qrels_file=qrels_path).load_custom()

results = json.load(open(f'../output/{dataset}{subtask}_test_out.json', 'r'))

evaluation = EvaluateRetrieval()
metrics = evaluation.evaluate(qrels, results, [10, 1000])
mrr = evaluation.evaluate_custom(qrels, results, [len(corpus)], metric='mrr')
scores = {'MRR': mrr[f'MRR@{len(corpus)}'], 'P@10': metrics[3]['P@10'], \
		  'NDCG@10': metrics[0]['NDCG@10'], 'R@1k': metrics[2]['Recall@1000']}
print(scores)

with open('scores.txt', 'a') as fout:
	fout.write('{:.2f}'.format(scores['MRR']*100)+'\t'+'{:.2f}'.format(scores['P@10']*100)+'\t'+ \
			   '{:.2f}'.format(scores['NDCG@10']*100)+'\t'+'{:.2f}'.format(scores['R@1k']*100)+'\n')
