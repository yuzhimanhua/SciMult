dataset=MAG
model=scimult_moe.ckpt
python3.8 eval_classification_coarse.py --dataset ${dataset} --model ${model}

dataset=MeSH
model=scimult_moe.ckpt
python3.8 eval_classification_coarse.py --dataset ${dataset} --model ${model}
