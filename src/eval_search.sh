dataset=Search
model=scimult_moe.ckpt
python3.8 eval_search.py --dataset ${dataset} --model ${model}


dataset=TRECCOVID_SciRepEval
model=scimult_moe.ckpt
python3.8 eval_search.py --dataset ${dataset} --model ${model}


dataset=TRECCOVID_BEIR
model=scimult_moe.ckpt
python3.8 eval_search.py --dataset ${dataset} --model ${model}


dataset=SciFact
model=scimult_moe.ckpt
python3.8 eval_search.py --dataset ${dataset} --model ${model}


dataset=NFCorpus
model=scimult_moe.ckpt
python3.8 eval_search.py --dataset ${dataset} --model ${model}
