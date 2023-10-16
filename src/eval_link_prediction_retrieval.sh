dataset=SciDocs
model=scimult_moe.ckpt
python3.8 eval_link_prediction_retrieval.py --dataset ${dataset} --model ${model}

dataset=PMCPatientsPPR
model=scimult_moe.ckpt
python3.8 eval_link_prediction_retrieval.py --dataset ${dataset} --model ${model}
