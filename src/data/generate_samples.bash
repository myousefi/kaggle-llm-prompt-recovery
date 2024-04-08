#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --job-name=GSample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --array=1-100%6

#SBATCH -o out/output_%A_%a.txt
#SBATCH -e out/error_%A_%a.txt

python generate_samples_pipeline.py