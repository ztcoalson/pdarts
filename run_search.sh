#!/bin/bash
#SBATCH -J pdarts
#SBATCH -A sail
#SBATCH -p sail
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 0-12:00:00
#SBATCH --output=./logs/slurm-%j.out

module load cuda/11.2

python train_search.py \
       --tmp_data_dir ../data/ \
       --save ./results/ \
       --add_layers 6 \
       --add_layers 12 \
       --dropout_rate 0.1 \
       --dropout_rate 0.4 \
       --dropout_rate 0.7 \
       --note smartlf1 \
       --poisons_type label_flip \
       --poisons_path "/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/smart-lf/smart-lf-resnet18-cifar10-1.0%.pth" \
       # --save_full_model \