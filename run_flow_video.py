import os
import torch
import numpy as np
import argparse
import cv2
import glob
import matplotlib.pyplot as plt

from flowlib import visualize_flow

from models import FlowNet2  # the path is depended on where you create this module

import warnings
warnings.filterwarnings("ignore")


def load_image_pair(path1, path2, dest_size=None):
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    if dest_size is not None:
        assert isinstance(dest_size, (list, tuple)) and len(dest_size) == 2
        img1 = cv2.resize(img1, dest_size)
        img2 = cv2.resize(img2, dest_size)
    return img1, img2


def load_video_frames(in_path, dest_size=None, save_file=False):
    # convert frames video file to pt file
    def load_video(file, im_size=None):
        imgs = []
        cap = cv2.VideoCapture(file)
        if not cap.isOpened():
            print("Error opening file", file)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if im_size is not None:
                    frame = cv2.resize(frame, im_size)
                imgs.append(frame)
            else:
                break
        cap.release()
        return imgs

    # convert images in a directory to pt file
    def load_imgs_in_directory(path, ext, im_size=None):
        files = sorted(glob.glob(path + "/*." + ext))
        if im_size is not None:
            imgs = [cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), im_size) for file in files]
        else:
            imgs = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]
        return imgs

    root, ext = os.path.splitext(in_path)
    if ext == ".npy":
        data = np.load(in_path)
        unique_shapes = np.unique([datum.shape for datum in data], axis=0)
        assert len(unique_shapes) == 1
        if dest_size is not None:
            assert data[0].shape[:2] == dest_size[::-1]
        return data

    # out_file = root + "_images.npy"
    # if os.path.exists(out_file):
    #     data = np.load(out_file)
    #     unique_shapes = np.unique([datum.shape for datum in data], axis=0)
    #     assert len(unique_shapes) == 1
    #     if dest_size is not None:
    #         assert data[0].shape[:2] == dest_size[::-1]
    #     return data

    if ext in (".avi", ".mp4"):   # process new data from video
        data = load_video(in_path, im_size=dest_size)
    elif len(ext) == 0:             # process new data from images in directory
        data = load_imgs_in_directory(in_path, "*", im_size=dest_size)
    else:
        print("Unknown data type:", in_path)
        return None

    data = np.array(data).astype(np.float32)

    # if save_file:
    #     np.save(out_file, data)
    return data


if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_false", help="Run model in pseudo-fp16 mode (fp16 storage fp32 math).")
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument("--in_path", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--recalc", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--visualize", type=int, default=0)

    args = parser.parse_args()
    args.fp16 = False

    # initial FlowNet2 and load pretrained weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = FlowNet2(args)
    net.to(device)
    # load the state_dict
    dict = torch.load("./models/FlowNet2_checkpoint.pth")
    net.load_state_dict(dict["state_dict"])

    # load the image pair
    scale = args.scale
    base_size = (args.width, args.height)
    upscaled_size = (scale * base_size[0], scale * base_size[1])

    if args.in_path is None:
        img1, img2 = load_image_pair("./test/img1.tif", "./test/img2.tif", dest_size=base_size)

        images = [img1, img2]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).to(device)

        # process the image pair to obtian the flow
        result = net(im).squeeze()

        data = result.data.cpu().numpy().transpose(1, 2, 0)

        print("input shape:", im.shape)
        print("data shape:", data.shape)
        if args.visualize != 0:
            visualize_flow(data)
    else:
        if os.path.exists(args.out_file) and args.recalc == 0:
            print("File existed -> skip!!!")
        else:
            data = load_video_frames(args.in_path, dest_size=None)
            n_frame = len(data) - 1  # -1 because we consider pairs of frames
            outflows = []
            for i in range(0, n_frame, args.batch):
                # determine frame indices in batch
                idx0, idx1 = i, min(i + args.batch, n_frame)
                # forming data batch
                frames_first = data[idx0:idx1]
                frames_second = data[idx0+1:idx1+1]
                if scale > 1:
                    frames_first = [cv2.resize(frame, upscaled_size) for frame in frames_first]
                    frames_second = [cv2.resize(frame, upscaled_size) for frame in frames_second]
                pairs = list(zip(frames_first, frames_second))
                # transpose data and feed into network
                pairs = np.array([np.array(pair).transpose(3, 0, 1, 2) for pair in pairs])
                im = torch.from_numpy(pairs.astype(np.float32)).to(device)
                # get out optical flow
                flows = net(im)
                outflows += [cv2.resize(flow.data.cpu().numpy().transpose(1, 2, 0), base_size) for flow in flows]
            outflows += [np.zeros_like(outflows[-1])]
            outflows = np.array(outflows).astype(np.float32)
            assert len(outflows) == len(data)
            data = np.array([cv2.resize(img, base_size) for img in data])
            print("imgs:", data.shape)
            print("flows:", outflows.shape)

            # concatenate data and save (n, h, w, c)
            outdata = np.concatenate((data, outflows), axis=3)
            np.save(args.out_file, outdata)
            print("output data:", outdata.shape)

            # visualizing only for checking
            if args.visualize != 0:
                data = np.load(args.out_file)
                print("data shape:", data.shape)
                plt.imshow(data[0, :, :, :3] / 255.)
                plt.show()
                visualize_flow(data[0, :, :, 3:])
                plt.imshow(data[-2, :, :, :3] / 255.)
                plt.show()
                visualize_flow(data[-2, :, :, 3:])
