dataset=Recommendation
model=scimult_moe.ckpt
python3.8 eval_link_prediction_reranking.py --dataset ${dataset} --model ${model}
