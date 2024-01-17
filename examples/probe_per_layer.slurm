#!/bin/bash
#SBATCH --cluster=wice
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=54
#SBATCH --gpus-per-node=3
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=bert_probe

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-env

cd $SLURM_SUBMIT_DIR

layer=$3
pooling=$2 #mean,cls
lang=$1 #0,1,2,3,4

python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer $layer
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer $layer
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer $layer







