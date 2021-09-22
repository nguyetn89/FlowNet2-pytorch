import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm

def video2frames(
    video_path: str,
    output_path: str,
    max_clip_length: int,
    clip_idx: int,
    z_length: int
) -> tuple:
    """Convert video to directories of frames

    Args:
        video_path (str): path to video
        video_path (str): output path to generate directory of frames
        max_clip_length (int): maximum allowed number of frames
        clip_idx (int): index to generate directory name
        z_length (int): length to zero-pad clip index

    Returns:
        tuple: (video file name [str], created directories [list])
    """

    video_file_name = os.path.split(video_path)[-1]

    if not os.path.exists(video_path):
        return video_file_name, []

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    video_reader = cv2.VideoCapture(video_path)
    n_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    n_clips = int(np.ceil(n_frames / max_clip_length))
    n_frames_per_clip = n_frames // n_clips

    n_extracted_frames = 0
    gen_dirs = []

    for i in range(n_clips):
        tmp_clip_idx = clip_idx + i
        dir_path = f"{output_path}/clip_{str(tmp_clip_idx).zfill(z_length)}"
        gen_dirs.append(dir_path)
        os.mkdir(dir_path)
        tmp_frame_count = 0
        while (tmp_frame_count < n_frames_per_clip or i == n_clips - 1) and n_extracted_frames < n_frames:
            retrieved, frame = video_reader.read()
            if not retrieved:
                break
            # save image to file
            cv2.imwrite(f"{dir_path}/frame_{str(tmp_frame_count).zfill(4)}.png", frame)
            tmp_frame_count += 1
            n_extracted_frames += 1
        print(f"Saved frames to {dir_path}")
    video_reader.release()
    return video_file_name, gen_dirs


def convert_videos(
    video_dir: str,
    video_ext: str,
    output_dir: str,
    max_clip_length: int,
    z_length: int
) -> int:
    """Convert videos in a directory to directories of frames

    Args:
        video_dir (str): directory path containing videos
        video_ext (str): extension of video files
        output_dir (str): directory path storing extracted directories of frames
        max_clip_length (int): maximum allowed number of frames
        z_length (int): length to zero-pad clip index
    
    Returns:
        int: number of created directory
    """

    video_files = sorted(glob.glob(f"{video_dir}/*.{video_ext}"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    clip_idx = 0
    generated_clips = ""
    progress_bar = tqdm(video_files)
    for video_file in progress_bar:
        video_file_name, generated_dirs = \
            video2frames(video_file, output_dir, 7500, clip_idx, z_length)
        generated_clips += "\n".join(
            f"{video_file_name}, {generated_dir}" \
                for generated_dir in generated_dirs
        ) + "\n"
        clip_idx += len(generated_dirs)
    
    with open(f"{output_dir}/_name_mapping.csv", 'w') as writer:
        writer.write(generated_clips)
    return clip_idx


def video2clips(video_path: str, max_clip_length: int) -> list:

    if not os.path.exists(video_path):
        return []
    
    dir_path, video_file_name = os.path.split(video_path)

    video_reader = cv2.VideoCapture(video_path)
    n_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_frames <= max_clip_length:
        return []

    n_clips = int(np.ceil(n_frames / max_clip_length))
    n_frames_per_clip = n_frames // n_clips

    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

    n_processed_frames = 0
    gen_videos = []

    for i in range(n_clips):
        out_video = f"{dir_path}/{video_file_name.split('.')[0]}_x{i}.{video_file_name.split('.')[-1]}"
        gen_videos.append(out_video)
        video_writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
        tmp_frame_count = 0
        while (tmp_frame_count < n_frames_per_clip or i == n_clips - 1) and n_processed_frames < n_frames:
            retrieved, frame = video_reader.read()
            if not retrieved:
                break
            # save image to file
            video_writer.write(frame)
            tmp_frame_count += 1
            n_processed_frames += 1

    video_reader.release()
    video_writer.release()
    os.rename(video_path, f"{video_path}.bak")

    return gen_videos


def split_videos(video_dir: str, video_ext: str, max_clip_length: int) -> list:
    video_files = sorted(glob.glob(f"{video_dir}/*.{video_ext}"))
    
    processed_videos = []
    progress_bar = tqdm(video_files)
    for video_file in progress_bar:
        processed_videos += video2clips(video_file, max_clip_length)
    return processed_videos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="path to video directory", type=str, default=None)
    parser.add_argument("--output_dir", help="path to output directory", type=str, default=None)

    # parsing
    args = parser.parse_args()

    # process
    print("START >>>>>>>>>>>>>>>>>>>>>")
    if args.output_dir is not None:
        n_dirs = convert_videos(args.video_dir, "mp4", args.output_dir, 7500, 5)
        print(f"Generated {n_dirs} directories of frames")
    else:
        processed_videos = split_videos(args.video_dir, "mp4", 7500)
        for i, vid in enumerate(processed_videos):
            print(f"[{str(i).zfill(3)}] {vid}")
    print("DONE <<<<<<<<<<<<<<<<<<<<<<")
