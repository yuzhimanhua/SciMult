model=biencoder_mod2attn_n3_taser_encoder_w_random_pos_data_v4_full_neg_3xG16_from_pubmedbert_abs
ckpt=9

dataset=paper_metadata_mag_mesh.json
expert=entity-retrieval
out_fn=cls.jsonl
python3.8 eval_scidocs.py --dataset ${dataset} --model ${model} --ckpt ${ckpt} --expert ${expert} --out_fn ${out_fn}

dataset=paper_metadata_view_cite_read.json
expert=multi-hop-retrieval
out_fn=user-citation.jsonl
python3.8 eval_scidocs.py --dataset ${dataset} --model ${model} --ckpt ${ckpt} --expert ${expert} --out_fn ${out_fn}
