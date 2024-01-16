#!/bin/bash

#layers=(1 2 3 4 5 6 7 8 9 10 11 12 'all')
task=$1
pooling=$2
layers=$3

#for layer in ${layers[@]}; do
 #       python transformers_bert.py --model_name bert --task_index $task --pooling $pooling --layer $layer
#        wait
#done

python transformers_pixel.py --model_name pixel --task_index $task --pooling $pooling --layer $layers

