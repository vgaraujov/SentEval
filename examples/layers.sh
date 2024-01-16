#!/bin/bash

#layers=(1 2 3 4 5 6 7 8 9 10 11 12 'all')
lang=$1 #1_Arabic, 2_Chinese, 3_Hebrew, 4_Hindi, 5_Russian, 6_Tamil
pooling=$2
layers=$3

#for layer in ${layers[@]}; do
 #       python transformers_bert.py --model_name bert --task_index $task --pooling $pooling --layer $layer
#        wait
#done

python transformers_bert.py --model_name mbert --language $lang --pooling $pooling --layer $layers

