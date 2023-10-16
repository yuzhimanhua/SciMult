dataset=CSConference
model=scimult_moe.ckpt
python3.8 eval_classification_fine.py --dataset ${dataset} --model ${model}

dataset=ChemistryMeSH
model=scimult_moe.ckpt
python3.8 eval_classification_fine.py --dataset ${dataset} --model ${model}

dataset=Geography
model=scimult_moe.ckpt
python3.8 eval_classification_fine.py --dataset ${dataset} --model ${model}

dataset=Psychology
model=scimult_moe.ckpt
python3.8 eval_classification_fine.py --dataset ${dataset} --model ${model}
