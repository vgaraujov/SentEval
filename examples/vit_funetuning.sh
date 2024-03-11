#!/bin/bash

source activate pixel-new

export WANDB_API_KEY=$2
export WANDB_PROJECT="vit-mae-finetuning"
export TREEBANK=$1
export DATA_DIR="../data/${TREEBANK}"
export FALLBACK_FONTS_DIR="fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="facebook/vit-mae-base" # also works with "bert-base-cased", "roberta-base", etc.
export SEQ_LEN=196
export BSZ=64
export GRAD_ACCUM=1
export LR=5e-5
export SEED=42
export NUM_STEPS=15000
export CONFIG="Team-PIXEL/pixel-base"

export RUN_NAME="${TREEBANK}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

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