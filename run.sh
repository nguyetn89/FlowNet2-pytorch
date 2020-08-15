#!/bin/bash
#./execute.sh UCSDped2 both
#./execute.sh Avenue both
#python3 split_video_sets.py --video "/home/nguyetn/projects/def-jeandiro/nguyetn/datasets/Traffic-Belleview/input.avi" --training_frames 300 --test_frames 300
#python3 split_video_sets.py --video "/home/nguyetn/projects/def-jeandiro/nguyetn/datasets/Traffic-Train/input.avi" --training_frames 0-800 --test_frames 13840-18000
./execute.sh Belleview both
./execute.sh Train both
