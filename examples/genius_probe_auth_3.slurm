#!/bin/bash
#SBATCH --clusters=genius
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p100
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=genuis_probe_auth

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-env

cd $SLURM_SUBMIT_DIR

layers=(1 2 3 4 5 6 7 8 9 10 11 12 'all')
#pooling=$2 #mean,cls
lang=$1
auth = ""

if [ $lang == "English"]; then
  for layer in ${layers[@]};
  do
    python transformers_pixel.py --model_name mpixel --language $lang --pooling mean --layer $layer --task_index 4 --auth $auth
  done
else
  for layer in ${layers[@]};
  do
    python transformers_pixel.py --model_name mpixel --language $lang --pooling mean --layer $layer --auth $auth
  done
fi
