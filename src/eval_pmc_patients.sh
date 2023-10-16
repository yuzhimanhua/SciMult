dataset=PMCPatients
subtask=PAR
model=scimult_moe_pmcpatients_par.ckpt
python3.8 eval_pmc_patients.py --dataset ${dataset} --subtask ${subtask} --model ${model}
python3.8 calc_pmc_patients_score.py --dataset ${dataset} --subtask ${subtask}

subtask=PPR
model=scimult_moe_pmcpatients_ppr.ckpt
python3.8 eval_pmc_patients.py --dataset ${dataset} --subtask ${subtask} --model ${model}
python3.8 calc_pmc_patients_score.py --dataset ${dataset} --subtask ${subtask}
