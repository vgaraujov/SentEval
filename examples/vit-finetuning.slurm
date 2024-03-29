#!/bin/bash
#SBATCH --clusters=genius
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p100
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=vit-finetune

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate pixel-new

cd $SLURM_SUBMIT_DIR

WANDB_API_KEY=$3
WANDB_PROJECT="vit-mae-finetuning"
DATA=$2
DATA_DIR="../data/${DATA}"
FALLBACK_FONTS_DIR="fallback_fonts"  # let's say this is where we downloaded the fonts to
MODEL="facebook/vit-mae-base" # also works with "bert-base-cased", "roberta-base", etc.
SEQ_LEN=196
BSZ=64
GRAD_ACCUM=1
LR=5e-5
SEED=42
NUM_STEPS=15000
CONFIG="Team-PIXEL/pixel-base"
TASK=$1


RUN_NAME="${DATA}-${TASK}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"


if [[ "$TASK" == "pos" ]]; then
  python run_pos.py \
    --model_name_or_path=${MODEL} \
    --processor_name=${CONFIG} \
    --remove_unused_columns=False \
    --data_dir=${DATA_DIR} \
    --do_train \
    --do_eval \
    --do_predict \
    --dropout_prob=0.1 \
    --max_seq_length=${SEQ_LEN} \
    --max_steps=${NUM_STEPS} \
    --num_train_epochs=10 \
    --early_stopping \
    --early_stopping_patience=5 \
    --per_device_train_batch_size=${BSZ} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --learning_rate=${LR} \
    --warmup_steps=100 \
    --run_name=${RUN_NAME} \
    --output_dir=${RUN_NAME} \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy=steps \
    --logging_steps=100 \
    --evaluation_strategy=steps \
    --eval_steps=500 \
    --save_strategy=steps \
    --save_steps=500 \
    --save_total_limit=5 \
    --report_to=wandb \
    --log_predictions \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_accuracy" \
    --fp16 \
    --half_precision_backend=apex \
    --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
    --seed=${SEED}

elif [[ "$TASK" == "ner" ]]; then
  python run_ner.py \
    --model_name_or_path=${MODEL} \
    --processor_name=${CONFIG} \
    --remove_unused_columns=False \
    --data_dir=${DATA_DIR} \
    --do_train \
    --do_eval \
    --do_predict \
    --dropout_prob=0.1 \
    --max_seq_length=${SEQ_LEN} \
    --max_steps=${NUM_STEPS} \
    --num_train_epochs=10 \
    --early_stopping \
    --early_stopping_patience=5 \
    --per_device_train_batch_size=${BSZ} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --learning_rate=${LR} \
    --warmup_steps=100 \
    --run_name=${RUN_NAME} \
    --output_dir=${RUN_NAME} \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy=steps \
    --logging_steps=100 \
    --evaluation_strategy=steps \
    --eval_steps=500 \
    --save_strategy=steps \
    --save_steps=500 \
    --save_total_limit=5 \
    --report_to=wandb \
    --log_predictions \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_f1" \
    --fp16 \
    --half_precision_backend=apex \
    --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
    --seed=${SEED}



elif [[ "$TASK" == "dep" ]]; then
  python run_ud.py \
    --model_name_or_path=${MODEL} \
    --processor_name=${CONFIG} \
    --remove_unused_columns=False \
    --data_dir=${DATA_DIR} \
    --do_train \
    --do_eval \
    --do_predict \
    --dropout_prob=0.1 \
    --max_seq_length=${SEQ_LEN} \
    --max_steps=${NUM_STEPS} \
    --num_train_epochs=10 \
    --early_stopping \
    --early_stopping_patience=5 \
    --per_device_train_batch_size=${BSZ} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --learning_rate=${LR} \
    --warmup_steps=100 \
    --run_name=${RUN_NAME} \
    --output_dir=${RUN_NAME} \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy=steps \
    --logging_steps=100 \
    --evaluation_strategy=steps \
    --eval_steps=500 \
    --save_strategy=steps \
    --save_steps=500 \
    --save_total_limit=5 \
    --report_to=wandb \
    --log_predictions \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_las" \
    --fp16 \
    --half_precision_backend=apex \
    --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
    --seed=${SEED}
fi
