#!/bin/bash
#SBATCH --partition=special
#SBATCH --job-name=regjob
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=%N-%j_d_MPTNET_SYNAPSE_CONFIG1.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
source env_lightning2/bin/activate
python lightning_train.py --dataset ACDC --exp EXP_MPTNET_SYNAPSE_CONFIG1_1
