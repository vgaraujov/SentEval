#!/bin/bash
#SBATCH --cluster=wice
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=16:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=probe_auth

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-env

cd $SLURM_SUBMIT_DIR

layers=(10 11 12 'all')
lang=$1
auth="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"

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
