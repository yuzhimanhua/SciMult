dataset=paper_metadata_mag_mesh.json
model=scimult_moe.ckpt
expert=entity-retrieval
out_fn=cls.jsonl
python3.8 eval_scidocs.py --dataset ${dataset} --model ${model} --expert ${expert} --out_fn ${out_fn}

dataset=paper_metadata_view_cite_read.json
model=scimult_moe.ckpt
expert=multi-hop-retrieval
out_fn=user-citation.jsonl
python3.8 eval_scidocs.py --dataset ${dataset} --model ${model} --expert ${expert} --out_fn ${out_fn}
