import os
import argparse
import cv2
import numpy as np


Exit = {"train": [[0, 7500]],
        "test": [[9680, 9710], [12370, 12500], [14020, 14070], [14760, 14816], [15290, 15500],
                 [15815, 15945], [17340, 17770], [18150, 18970], [18970, 19450], [19875, 19950],
                 [20170, 20225], [20560, 20738], [21760, 21805], [22830, 23610], [24020, 24255],
                 [37185, 37267], [37940, 38005], [40710, 41750], [47020, 47505], [47670, 47910],
                 [49230, 49280], [50365, 50680], [50940, 51295], [52230, 52395], [52765, 53050],
                 [53330, 53425], [54375, 54475], [55315, 56065], [57120, 57615], [59110, 62195]]}


Entrance = {"train": [[1560, 2100], [2650, 4120], [7650, 7755], [8220, 9030], [9295, 9535],
                      [10585, 11350], [12475, 12820], [13965, 14650], [16530, 17000], [17735, 19510],
                      [19515, 20130], [21950, 26140]],
            "test": [[2175, 2650], [4120, 5560], [12820, 13025], [16120, 16530], [17020, 17735],
                     [20130, 20290], [20290, 20770], [21650, 21865], [27500, 29950], [29950, 31080],
                     [31340, 31425], [32130, 32760], [32955, 32990], [33240, 35455], [35500, 36350],
                     [36370, 36480], [36500, 37055], [37610, 37710], [38120, 39390], [39390, 39595],
                     [39980, 40050], [40215, 40400], [40435, 40575], [40600, 41970], [42085, 42485],
                     [44845, 48840], [49095, 49245], [49510, 49560], [52100, 55500], [55595, 57585],
                     [57700, 57800], [58680, 59730], [59800, 60980], [61980, 67350], [67350, 68200],
                     [68470, 69350], [69520, 70230], [70645, 70745], [70770, 72200], [72200, 72835],
                     [72880, 73140], [73360, 74110], [74260, 77076], [79300, 79365], [79420, 80510],
                     [80900, 80970], [81130, 81770], [82210, 82415], [82490, 84475], [84475, 89170],
                     [89570, 89920], [90000, 94220], [94255, 100700], [101200, 102530], [105975, 111300],
                     [111300, 111645], [111745, 113050], [114895, 116040], [116110, 116580], [117465, 117960],
                     [118020, 118110], [118135, 119130], [119185, 120180], [120430, 121285], [123280, 123885],
                     [124350, 124445], [124645, 125240], [127220, 127775], [127925, 128475], [130225, 130760],
                     [133160, 133660], [134660, 135410], [137160, 137965], [138610, 139315], [140610, 141890],
                     [142740, 143540]]}


def get_clip_idx(frame, clips):
    for idx, clip in enumerate(clips):
        if frame in range(clip[0], clip[1]):
            return idx
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="path to video file", type=str, default=None)
    parser.add_argument("--dataset", help="Entrance or Exit", type=str, default=None)
    parser.add_argument("--only", help="train, test, or both (default) sets", type=str, default="both")

    # parsing
    args = parser.parse_args()
    assert args.dataset in ('Entrance', 'Exit')
    assert args.only in ("train", "test", "both")
    directory = os.path.dirname(args.video)
    print("Dataset: %s, set: %s" % (args.dataset, args.only))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error opening video", args.video)
    dataset = Exit if args.dataset == "Exit" else Entrance

    # check frame range setting
    max_clip_length = 7500
    assert np.all([x[1] - x[0] <= max_clip_length for x in dataset["train"]])
    assert np.all([x[1] - x[0] <= max_clip_length for x in dataset["test"]])

    # create directories for training data
    n_training_directories = len(dataset["train"])
    training_paths = []
    for i in range(n_training_directories):
        training_paths.append("%s/train/%s" % (directory, str(i+1).zfill(3)))
        if not os.path.exists(training_paths[-1]):
            os.makedirs(training_paths[-1])
    saved_training_frame = 0

    # create directories for test data
    n_test_directories = len(dataset["test"])
    test_paths = []
    for i in range(n_test_directories):
        test_paths.append("%s/test/%s" % (directory, str(i+1).zfill(3)))
        if not os.path.exists(test_paths[-1]):
            os.makedirs(test_paths[-1])
    saved_test_frame = 0

    # read video and save to images
    z_length = 6
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # training frame
            if args.only != "test":
                clip_idx = get_clip_idx(frame_idx, dataset["train"])
                if clip_idx is not None:
                    out_file = "%s/%s.png" % (training_paths[clip_idx], str(frame_idx).zfill(z_length))
                    cv2.imwrite(out_file, frame)
                    saved_training_frame += 1
            # test frame
            if args.only != "train":
                clip_idx = get_clip_idx(frame_idx, dataset["test"])
                if clip_idx is not None:
                    out_file = "%s/%s.png" % (test_paths[clip_idx], str(frame_idx).zfill(z_length))
                    cv2.imwrite(out_file, frame)
                    saved_test_frame += 1
            frame_idx += 1
        else:
            break
    cap.release()
    print("Done!!!")
    print("Expect: %d training and %d test frames"
          % (np.sum([x[1] - x[0] for x in dataset["train"]]), np.sum([x[1] - x[0] for x in dataset["test"]])))
    print("Saved:  %d training and %d test frames" % (saved_training_frame, saved_test_frame))
