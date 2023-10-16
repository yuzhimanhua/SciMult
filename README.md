# Pre-training Multi-task Contrastive Learning Models for Scientific Literature Understanding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code and instructions for reproducing the experiments in the paper
[Pre-training Multi-task Contrastive Learning Models for Scientific Literature Understanding](https://arxiv.org/abs/2305.14232) (Findings of EMNLP 2023).

## Links
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Models](#models)
- [Citation](#citation)

## Installation
We use one NVIDIA RTX A6000 GPU to run the evaluation code in our experiments. The code is written in Python 3.8. You can install the dependencies as follows.
```bash
git clone --recurse-submodules https://github.com/yuzhimanhua/SciMult
cd SciMult
conda env create --file=environment.yml --name=scimult

# activate the new sandbox you just created
conda activate scimult
# add the `src/` and `third_party/DPR` to the list of places python searches for packages
conda develop src/ third_party/DPR/

# download spacy models
python -m spacy download en_core_web_sm
```

## Quick Start
You need to first download the [evaluation datasets](https://drive.google.com/file/d/1hoUAInDVO_UYnQiOOoVuBjwnVgY0BosO/view?usp=drive_link) and the [pre-trained models](https://huggingface.co/yuz9yuz/SciMult/tree/main). After you unzip the dataset file, put the folder (i.e., ```data/```) under the repository main folder ```./```. After you download the four model checkpoints (i.e., ```scimult_vanilla.ckpt```, ```scimult_moe.ckpt```, ```scimult_moe_pmcpatients_par.ckpt```, and ```scimult_moe_pmcpatients_ppr.ckpt```), put them under the model folder ```./model/```.

Then, you can run the evaluation code for each task:
```bash
cd src

# evaluate fine-grained classification (MAPLE [CS-Conference, Chemistry-MeSH, Geography, Psychology])
./eval_classification_fine.sh

# evaluate coarse-grained classification (SciDocs [MAG, MeSH])
./eval_classification_coarse.sh

# evaluate link prediction under the retrieval setting (SciDocs [Cite, Co-cite], PMC-Patients [PPR])
./eval_link_prediction_retrieval.sh

# evaluate link prediction under the reranking setting (Recommendation)
./eval_link_prediction_reranking.sh

# evaluate search (SciRepEval [Search, TREC-COVID], BEIR [TREC-COVID, SciFact, NFCorpus])
./eval_search.sh
```
The metrics will be shown at the end of the terminal output as well as in ```scores.txt```.

### PMC-Patients
To reproduce our performance on the [PMC-Patients Leaderboard](https://pmc-patients.github.io/):
```bash
cd src
./eval_pmc_patients.sh
```
The metrics will be shown at the end of the terminal output as well as in ```scores.txt```. The similarity scores that we submitted to the leaderboard can be found at ```../output/PMCPatientsPAR_test_out.json``` and ```../output/PMCPatientsPPR_test_out.json```.

### SciDocs
To reproduce our performance on the [SciDocs](https://github.com/allenai/scidocs) benchmark:
```bash
cd src
./eval_scidocs.sh
```
The output embedding files can be found at ```../output/cls.jsonl``` and ```../output/user-citation.jsonl```. Then, run the adapted SciDocs evaluation code:
```bash
cd ../
git clone https://github.com/yuzhimanhua/SciDocs.git
cd scidocs

# install dependencies
conda deactivate
conda create -y --name scidocs python==3.7
conda activate scidocs
conda install -y -q -c conda-forge numpy pandas scikit-learn=0.22.2 jsonlines tqdm sklearn-contrib-lightning pytorch
pip install pytrec_eval awscli allennlp==0.9 overrides==3.1.0
python setup.py install

# run evaluation
python eval.py
```
The metrics will be shown at the end of the terminal output.

## Datasets
TBD

## Models
TBD

## Citation
If you find SciMult useful in your research, please cite the following paper:
```
@article{zhang2023pre,
  title={Pre-training Multi-task Contrastive Learning Models for Scientific Literature Understanding},
  author={Zhang, Yu and Cheng, Hao and Shen, Zhihong and Liu, Xiaodong and Wang, Ye-Yi and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2305.14232},
  year={2023}
}
```
