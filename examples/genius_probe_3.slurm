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
#SBATCH --job-name=genuis_probe

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-env

cd $SLURM_SUBMIT_DIR

layers=(1 2 3 4 5 6 7 8 9 10 11 12 'all')
#pooling=$2 #mean,cls
lang=$1
model=$2
seed=$3

if [ $model == "pixel" ]; then
  script="transformers_pixel.py"
elif [ $model == "bert" ]; then
  script="transformers_bert.py"
elif [ $model == "vit-mae" ]; then
  script="transformers_pixel.py"
elif [ $model == "mbert" ]; then
  script="transformers_bert.py"
elif [ $model == "xlmr" ]; then
  script="transformers_bert.py"
elif [ $model == "pixel-hindi" ]; then
  script="transformers_pixel_finetuned.py"
else
  echo "Invalid model name"
  exit 1
fi

if [ $lang == "English" ]; then
  for layer in ${layers[@]};
  do
    python $script --model_name $model --language $lang --pooling mean --layer $layer --task_index 4 --seed $seed
  done
else
  for layer in ${layers[@]};
  do
    python $script --model_name $model --language $lang --pooling mean --layer $layer --seed $seed
  done
fi




