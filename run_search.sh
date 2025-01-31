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
       --note noise-diff-denoise4 \
       --poisons_type diffusion_denoise \
       --poisons_path '/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/diffusion_denoise/datasets/denoised/gc_cifar10/denoise_gaussian_noise/denoised_w_sigma_0.1.pt' \
       --dset cifar10 \
       # --track_grads \
       # --save_full_model \
