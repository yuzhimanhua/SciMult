dataset=MAG
model=biencoder_mod2attn_n3_taser_encoder_w_random_pos_data_v4_full_neg_3xG16_from_pubmedbert_abs
ckpt=9
python3.8 eval_classification_coarse.py --dataset ${dataset} --model ${model} --ckpt ${ckpt}

dataset=MeSH
model=biencoder_mod2attn_n3_taser_encoder_w_random_pos_data_v4_full_neg_3xG16_from_pubmedbert_abs
ckpt=9
python3.8 eval_classification_coarse.py --dataset ${dataset} --model ${model} --ckpt ${ckpt}
