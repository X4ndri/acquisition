#!/bin/bash

/home/ahmad/miniconda3/envs/daq2/bin/python /home/ahmad/repos/cammy_aq/cammy/cli.py run --camera-options /home/ahmad/repos/cammy_aq/camera_options.toml --save-engine frames --record

read -p "Press any key to exit..."
python 