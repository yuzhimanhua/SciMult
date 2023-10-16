# Task-Aware Specialization for Efficient and Robust Dense Retrieval for Open-Domain Question Answering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code and instructions for reproducing the experiments in the paper
[Task-Aware Specialization for Efficient and Robust Dense Retrieval for Open-Domain Question Answering](https://aclanthology.org/2023.acl-short.159/) (ACL 2023).

![Approach Overview](./assets/taser.png?raw=true)

## Introduction (WIP)

```bash
git clone --recurse-submodules https://github.com/microsoft/taser

conda env create --file=environment.yml --name=taser

# activate the new sandbox you just created
conda activate taser
# add the `src/` and `third_party/DPR` to the list of places python searches for packages
conda develop src/ third_party/DPR/

# download spacy models
python -m spacy download en_core_web_sm
```

See [worksheets/01-in-domain-evaluation](./worksheets/01-in-domain-evaluation/) for steps to run in-domain evaluation experiments with TASER models.

More details coming soon!

## Citation

If you use any source code or data included in this repo, please cite our paper.

```bib
@inproceedings{cheng-etal-2023-task,
    title = "Task-Aware Specialization for Efficient and Robust Dense Retrieval for Open-Domain Question Answering",
    author = "Cheng, Hao  and
      Fang, Hao  and
      Liu, Xiaodong  and
      Gao, Jianfeng",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.159",
    pages = "1864--1875",
}
```