#!/bin/bash
#SBATCH --gres=gpu:2       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:30
#SBATCH --output=./log_calc_%N-%j.out
module load python/3.6
source ~/ENV/bin/activate
#./execute.sh UCSDped2 both
#./execute.sh Avenue both
#python3 split_video_sets.py --video "/home/nguyetn/projects/def-jeandiro/nguyetn/datasets/Traffic-Belleview/input.avi" --training_frames 300 --test_frames 300
#python3 split_video_sets.py --video "/home/nguyetn/projects/def-jeandiro/nguyetn/datasets/Traffic-Train/input.avi" --training_frames 0-800 --test_frames 13840-18000
./execute.sh Belleview both
./execute.sh Train both