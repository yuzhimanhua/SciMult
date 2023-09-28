dataset=Search
model=biencoder_mod2attn_n3_taser_encoder_w_random_pos_data_v4_full_neg_3xG16_from_pubmedbert_abs
ckpt=9
python3.8 eval_search.py --dataset ${dataset} --model ${model} --ckpt ${ckpt}


dataset=TRECCOVID_SciRepEval
model=biencoder_mod2attn_n3_taser_encoder_w_random_pos_data_v4_full_neg_3xG16_from_pubmedbert_abs
ckpt=9
python3.8 eval_search.py --dataset ${dataset} --model ${model} --ckpt ${ckpt}


dataset=TRECCOVID_BEIR
model=biencoder_mod2attn_n3_taser_encoder_w_random_pos_data_v4_full_neg_3xG16_from_pubmedbert_abs
ckpt=9
python3.8 eval_search.py --dataset ${dataset} --model ${model} --ckpt ${ckpt}


dataset=SciFact
model=biencoder_mod2attn_n3_taser_encoder_w_random_pos_data_v4_full_neg_3xG16_from_pubmedbert_abs
ckpt=9
python3.8 eval_search.py --dataset ${dataset} --model ${model} --ckpt ${ckpt}


dataset=NFCorpus
model=biencoder_mod2attn_n3_taser_encoder_w_random_pos_data_v4_full_neg_3xG16_from_pubmedbert_abs
ckpt=9
python3.8 eval_search.py --dataset ${dataset} --model ${model} --ckpt ${ckpt}
