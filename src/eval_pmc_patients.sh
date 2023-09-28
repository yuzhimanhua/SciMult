# To run calc_pmc_patients_score.py, you need to install the beir package: https://pypi.org/project/beir/
# python3.6 -m pip install beir

dataset=PMCPatients
subtask=PAR
model=biencoder_shared_taser_encoder_w_random_pos_pmc_patient_par_3xG16_from_mod2attn_taser_ckpt9
ckpt=9
python3.8 eval_pmc_patients.py --dataset ${dataset} --subtask ${subtask} --model ${model} --ckpt ${ckpt}
python3.6 calc_pmc_patients_score.py --dataset ${dataset} --subtask ${subtask}

subtask=PPR
model=biencoder_shared_taser_encoder_w_random_pos_pmc_patient_ppr_3xG16_from_mod2attn_taser_ckpt9
ckpt=9
python3.8 eval_pmc_patients.py --dataset ${dataset} --subtask ${subtask} --model ${model} --ckpt ${ckpt}
python3.6 calc_pmc_patients_score.py --dataset ${dataset} --subtask ${subtask}
