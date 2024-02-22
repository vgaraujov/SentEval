#!/bin/bash

layers=(1 2 3 4 5 6 7 8 9 10 11 12 'all')
lang=$1 #1_Arabic, 2_Chinese, 3_Hebrew, 4_Hindi, 5_Russian, 6_Tamil
#layers=$3


#for layer in ${layers[@]}; do
#        python transformers_bert.py --model_name mt5 --language $lang --pooling mean --layer $layer
#        wait
#done

#python transformers_pixel.py --model_name mpixel --language $lang --pooling $pooling --layer $layers --auth $auth
python transformers_bert.py --model_name mt5 --language $lang --pooling mean --layer all