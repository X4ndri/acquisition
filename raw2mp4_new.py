# rawfile = "/home/ahmad/Desktop/behavioral_recordings/intercam/test/raw/-/test_raw_-_-_2024-04-11_19-23-09_v2_5_5_12/Lucid Vision Labs-HTP003S-001-240200702.dat"

rawfile = "/data_sp/intercam/test_shafiq___v2_300_300_12_20240423092906-012203/Lucid Vision Labs-HTP003S-001-240200702.dat"

import cv2
import argparse
import numpy as np
from pathlib import Path


def intensity_to_rgba(frame, minval=452, maxval=3065, colormap=cv2.COLORMAP_TURBO):
    new_frame = np.ones((frame.shape[0], frame.shape[1], 4))
    disp_frame = frame.copy().astype("float")
    disp_frame -= minval
    disp_frame[disp_frame < 0] = 0
    disp_frame /= np.abs(maxval - minval)
    disp_frame[disp_frame >= 1] = 1
    disp_frame *= 255
    bgr_frame = cv2.applyColorMap(disp_frame.astype(np.uint8), colormap)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    new_frame[:, :, :3] = rgb_frame
    new_frame = new_frame.astype(np.uint8)
    return new_frame


def stack2mp4(imgs, output_video_path, dims, fps=30, colormap=cv2.COLORMAP_INFERNO, minval=1200, maxval=2200):
    """
    Convert a sequence of images into an MP4 video.
    
    Args:
    - image_files (list): List of file paths to the input images.
    - output_video_path (str): Path to save the output video.
    - fps (int): Frames per second of the output video.
    """

    print(f'saving to {output_video_path}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, dims, isColor=True)

    for i, im in enumerate(imgs):
        img = intensity_to_rgba(im, minval=minval, maxval=maxval, colormap=colormap)[:,:,:3]
        out.write(img)


    out.release()


def dir2mp4(rawfiledir, dims = [640, 480], fps=30, minval=452, maxval=3065, colormap=cv2.COLORMAP_INFERNO, preset = None):
    
    if preset == 'flir':
        minval=25
        maxval = 200
    if preset == 'lucid':
        minval = 452
        maxval=3065

    stemname = Path(rawfiledir).stem
    output_dir = Path(rawfiledir).parent.joinpath(f'{stemname}.avi').as_posix()

    # get images
    depth_data = np.fromfile(rawfiledir, dtype=np.uint16)  # Assuming little-endian uint16
    depth_data =  depth_data.reshape([-1, *dims[::-1]])

    stack2mp4(imgs=depth_data, output_video_path=output_dir, fps=fps,dims=dims, minval=minval, maxval=maxval, colormap = colormap)


def main():
        print('starting')
        parser = argparse.ArgumentParser(description='Convert a directory of images into an .mp4')
        parser.add_argument('parent_dir', type=str, help='Path to the directory containing the images')
        parser.add_argument('--output_path', type=str, help='Specify output path', default=None)
        parser.add_argument('--preset', type=str, help='specify a preset for minval and maxval pseudocoloring', default=None)
        parser.add_argument('--fps', type=int, help="framerate to save at", default=30)
        parser.add_argument('--min_val', type=int, default=1200, help='Minimum value of the input range')
        parser.add_argument('--max_val', type=int, default=2200, help='Maximum value of the input range')

        args = parser.parse_args()

        dir2mp4(rawfiledir=args.parent_dir, preset=args.preset, fps=args.fps, minval=args.min_val, maxval=args.max_val)


if __name__ == '__main__':
    main()