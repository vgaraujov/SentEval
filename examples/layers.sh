#!/bin/bash

layers=(1 2 3 4 5 6 7 8 9 10 11 12 'all')
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


if [$lang == "English"]; then
  python $script --model_name $model --language $lang --pooling mean --layer $layer --task_index 4
else
  python $script --model_name $model --language $lang --pooling mean --layer $layer
fi
