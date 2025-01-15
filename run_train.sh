#!/bin/bash
#SBATCH -J pdarts
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH -c 4
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00
#SBATCH --output=./logs/slurm-%j.out

module load cuda/11.2

python train_cifar.py \
       --tmp_data_dir ../data \
       --auxiliary \
       --cutout \
       --save ./results/ \
       --note robot-noise4-50% \
       --arch ROBOT_NOISE_50P_4