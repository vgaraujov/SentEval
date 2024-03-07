#!/bin/bash
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_v100
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=genius_probe_per_layer

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-env

cd $SLURM_SUBMIT_DIR


#pooling=$2 #mean,cls
lang=$1
model=$2
layer=$3

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
else
  echo "Invalid model name"
  exit 1
fi


if [ $lang == "English"]; then
  python $script --model_name $model --language $lang --pooling mean --layer $layer --task_index 4
else
  python $script --model_name $model --language $lang --pooling mean --layer $layer
fi




