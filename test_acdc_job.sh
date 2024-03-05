#!/bin/bash
#SBATCH --partition=special
#SBATCH --job-name=testing
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%N-%jMPTNETACDC.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
source env_lightning2/bin/activate
python lightning_acdc_test.py --model MPTNET --ckpt "ckpt/EXP_MPTNET_ACDC_CONFIG1_96_122144188_emb144_LAST/last-v2.ckpt"
