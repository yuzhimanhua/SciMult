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
To reproduce our performance on the [SciDocs benchmark](https://github.com/allenai/scidocs):
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
The preprocessed evaluation datasets can be downloaded from [here](https://drive.google.com/file/d/1hoUAInDVO_UYnQiOOoVuBjwnVgY0BosO/view?usp=drive_link). The aggregate version is released under the [ODC-By v1.0 License](https://opendatacommons.org/licenses/by/1-0/). By downloading this version you acknowledge that you have read and agreed to all the terms in this license. 

Similar to Tensorflow [datasets](https://github.com/tensorflow/datasets) or Hugging Face's [datasets](https://github.com/huggingface/datasets) library, we just downloaded and prepared public datasets. We only distribute these datasets in a specific format, but we do not vouch for their quality or fairness, or claim that you have the license to use the dataset. It remains the user's responsibility to determine whether you as a user have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.

More details about each constituent dataset are as follows.
| Dataset | Folder | #Queries | #Candidates | Source | License |
| ------- | ------ | -------- | ----------- | ------ | ------- |
| MAPLE (CS-Conference) | ```classification_fine/``` | 261,781 | 15,808 | [Link](https://github.com/yuzhimanhua/MAPLE) | [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) |
| MAPLE (Chemistry-MeSH) | ```classification_fine/``` | 762,129 | 30,194 | [Link](https://github.com/yuzhimanhua/MAPLE) | [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) |
| MAPLE (Geography) | ```classification_fine/``` | 73,883 | 3,285 | [Link](https://github.com/yuzhimanhua/MAPLE) | [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) |
| MAPLE (Psychology) | ```classification_fine/``` | 372,954 | 7,641 | [Link](https://github.com/yuzhimanhua/MAPLE) | [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) |
| SciDocs (MAG Fields) | ```classification_coarse/``` | 25,001 | 19 | [Link](https://github.com/allenai/scidocs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| SciDocs (MeSH Diseases) | ```classification_coarse/``` | 23,473 | 11 | [Link](https://github.com/allenai/scidocs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| SciDocs (Cite) | ```link_prediction_retrieval/``` | 92,214 | 142,009 | [Link](https://github.com/allenai/scidocs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| SciDocs (Co-cite) | ```link_prediction_retrieval/``` | 54,543 | 142,009 | [Link](https://github.com/allenai/scidocs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| PMC-Patients (PPR, Zero-shot) | ```link_prediction_retrieval/``` | 100,327 | 155,151 | [Link](https://github.com/pmc-patients/pmc-patients) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| PMC-Patients (PAR, Supervised) | ```pmc_patients/``` | 5,959 | 1,413,087 | [Link](https://github.com/pmc-patients/pmc-patients) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| PMC-Patients (PPR, Supervised) | ```pmc_patients/``` | 2,812 | 155,151 | [Link](https://github.com/pmc-patients/pmc-patients) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| SciDocs (Co-view) | ```scidocs/``` | 1,000 | reranking, 29.98 for each query on average | [Link](https://github.com/allenai/scidocs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| SciDocs (Co-read) | ```scidocs/``` | 1,000 | reranking, 29.98 for each query on average | [Link](https://github.com/allenai/scidocs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| SciDocs (Cite) | ```scidocs/``` | 1,000 | reranking, 29.93 for each query on average | [Link](https://github.com/allenai/scidocs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| SciDocs (Co-cite) | ```scidocs/``` | 1,000 | reranking, 29.95 for each query on average | [Link](https://github.com/allenai/scidocs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Recommendation | ```link_prediction_reranking/``` | 137 | reranking, 16.28 for each query on average | [Link](https://github.com/akanakia/microsoft-academic-paper-recommender-user-study) | N/A |
| SciRepEval-Search | ```search/``` | 2,637 | reranking, 10.00 for each query on average | [Link](https://github.com/allenai/scirepeval) | [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) |
| TREC-COVID in SciRepEval | ```search/``` | 50 | reranking, 1386.36 for each query on average | [Link](https://github.com/allenai/scirepeval) | [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) |
| TREC-COVID in BEIR | ```search/``` | 50 | 171,332 | [Link](https://github.com/beir-cellar/beir) | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| SciFact | ```search/``` | 1,109 | 5,183 | [Link](https://github.com/beir-cellar/beir) | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) |
| NFCorpus | ```search/``` | 3,237 | 3,633 | [Link](https://github.com/beir-cellar/beir) | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

## Models
Our pre-trained models can be downloaded from [here](https://huggingface.co/yuz9yuz/SciMult/tree/main). Please refer to the Hugging Face [README](https://huggingface.co/yuz9yuz/SciMult/) for more details about the models. 

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
