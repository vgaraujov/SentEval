#!/bin/bash
#SBATCH --cluster=wice
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=36
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=pixel_probe


cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-env

cd $SLURM_SUBMIT_DIR

layers=$3 #1,2,3,4,5,6,7,8,9,10,11,12,all
pooling=$2 #mean,cls
task=$1 #0,1,2,3,4

python transformers_pixel.py --model_name pixel --task_index $task --pooling $pooling --layer $layers


