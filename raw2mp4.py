rawfile = "/home/ahmad/Desktop/behavioral_recordings/intercam/test/raw/-/test_raw_-_-_2024-04-11_19-23-09_v2_5_5_12/Lucid Vision Labs-HTP003S-001-240200702.dat"

import numpy as np
import cv2

# Load raw depth data
width = 640  # Width of each frame
height = 480  # Height of each frame
num_frames = 295  # Total number of frames
depth_data = np.fromfile(rawfile, dtype=np.uint16)  # Assuming little-endian uint16

# Reshape the raw data into frames
frames = depth_data.reshape(num_frames, height, width)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify codec (for AVI format)
fps = 30  # Frames per second
out = cv2.VideoWriter('depth_video_rgb.avi', fourcc, fps, (width, height))

# Define colormap (jet colormap)
def depth_to_rgb(depth_map):
    normalized_depth = depth_map / np.max(depth_map)
    colormap = cv2.applyColorMap(np.uint8(normalized_depth * 255), cv2.COLORMAP_JET)
    return colormap

# Write frames to video
for frame in frames:
    # Convert depth data to RGB image
    frame_rgb = depth_to_rgb(frame)
    out.write(frame_rgb)

# Release the VideoWriter object
out.release()