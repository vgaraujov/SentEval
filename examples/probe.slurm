#!/bin/bash
#SBATCH --cluster=wice
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=pixel_probe

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-env

cd $SLURM_SUBMIT_DIR

layers=(1 2 3 4 5 6 7 8 9 10 11 12 'all')
pooling=$2 #mean,cls
lang=$1

#for layer in ${layers[@]};
#do
#  python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer $layer
#  python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer $layer
#  python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer $layer
#done

python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 1
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 2
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 3
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 4
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 5
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 6
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 7
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 8
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 9
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 10
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 11
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer 12
python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer all

python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 1
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 2
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 3
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 4
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 5
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 6
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 7
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 8
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 9
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 10
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 11
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer 12
python transformers_bert.py --model_name bert --language $lang --pooling $pooling --layer all

#python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 1
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 2
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 3
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 4
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 5
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 6
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 7
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 8
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 9
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 10
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 11
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer 12
python transformers_pixel.py --model_name pixel --language $lang --pooling $pooling --layer all
