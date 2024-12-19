#!/bin/bash
#SBATCH -J pdarts
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00

python train_search.py \
       --tmp_data_dir ../data/ \
       --save ./results/ \
       --add_layers 6 \
       --add_layers 12 \
       --dropout_rate 0.1 \
       --dropout_rate 0.4 \
       --dropout_rate 0.7 \
       --note clean4