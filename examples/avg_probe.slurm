#!/bin/bash
#SBATCH --cluster=wice
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=54
#SBATCH --gpus-per-node=3
#SBATCH --partition=gpu
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=bert_probe_avg

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-env

cd $SLURM_SUBMIT_DIR

task=(0 1 2 3 4)

for tasks in ${task[@]}; do
	python transformers_bert.py --model_name bert --task_index $tasks --pooling cls --layer all &
	python transformers_bert.py --model_name bert --task_index $tasks --pooling mean --layer all
	wait

done
