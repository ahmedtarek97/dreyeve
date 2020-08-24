#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import sys
import os
sys.path.append("Model/")
from os.path import join
from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.io_helper import normalize, read_image


def visualize(filepath):
    small_size = (270, 480)
    output_dir = join(filepath,'jpg')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for frame in range(0 + 15 , 749):
        x_img = read_image(join(filepath,"Frames",'{}.jpg'.format(frame)), channels_first=False,
                       color_mode='BGR', dtype=np.uint8)
        #x_img = cv2.resize(x_img, dsize=(621, 1632))

        dreyevenet_p = normalize(np.squeeze(np.load(join(filepath,"npz",'{:d}.npz'.format(frame)))['arr_0']))
        dreyevenet_p = cv2.resize(cv2.cvtColor(dreyevenet_p, cv2.COLOR_GRAY2BGR), small_size[::-1])
        dreyevenet_p = cv2.resize(dreyevenet_p, dsize=(1920,1080))
        dreyevenet_p = cv2.applyColorMap(dreyevenet_p, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(x_img, 0.5, dreyevenet_p, 0.5, gamma=0)
        cv2.imwrite(join(output_dir,'{}.jpg'.format(frame)),blend)