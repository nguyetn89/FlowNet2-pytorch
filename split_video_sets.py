
import os
import argparse
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="path to video file", type=str, default=None)
    parser.add_argument("--training_frames", help="range of training frames (e.g. 800 or 0-800)", type=str, default=None)
    parser.add_argument("--test_frames", help="range of test frames (e.g. 800 or 800-1000)", type=str, default=None)

    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error opening video", args.video)

    # range of training frames
    if '-' in args.training_frames:
        vals = args.training_frames.split('-')
        training_frames = (int(vals[0]), int(vals[1]))
    else:
        training_frames = (0, int(args.training_frames))
    print("Training frames: %d to %d" % training_frames)

    # range of test frames
    if '-' in args.test_frames:
        vals = args.test_frames.split('-')
        test_frames = (int(vals[0]), int(vals[1]))
    else:
        test_frames = (int(args.test_frames), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Test frames: %d to %d" % test_frames)

    # create directories for training and test sets
    directory = os.path.dirname(args.video)
    training_path = "%s/train/001" % directory
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    test_path = "%s/test/001" % directory
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # read video and save to images
    z_length = len(str(test_frames[1]))
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if idx in range(training_frames[0], training_frames[1]):
                out_file = "%s/%s.png" % (training_path, str(idx).zfill(z_length))
                cv2.imwrite(out_file, frame)
            elif idx in range(test_frames[0], test_frames[1]):
                out_file = "%s/%s.png" % (test_path, str(idx).zfill(z_length))
                cv2.imwrite(out_file, frame)
            idx += 1
        else:
            break
    cap.release()
    print("Done!!!")
