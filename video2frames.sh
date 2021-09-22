#!/bin/bash

# python3 run_flow_video.py --in_path="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/UCF_Crime/tmp/anomaly_detection/clips/Train/Normal_Videos308_x264.mp4" --out_file="./long/test_long_vid.npy" --scale 3 --recalc 1 --auto_split 1

python3 video2frames.py --video_dir="/home/nguyetn/projects/def-jeandiro/nguyetn/datasets/UCF_Crime/tmp/anomaly_detection/clips/Train" #--output_dir="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/UCF_Crime/tmp/anomaly_detection/clips/Train/separated_clips"
python3 video2frames.py --video_dir="/home/nguyetn/projects/def-jeandiro/nguyetn/datasets/UCF_Crime/tmp/anomaly_detection/clips/Test" #--output_dir="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/UCF_Crime/tmp/anomaly_detection/clips/Test/separated_clips"
