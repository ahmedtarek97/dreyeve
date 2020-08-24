

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import argparse
import pyflow
import os
import cv2
from os.path import join

parser = argparse.ArgumentParser(
        description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
        '-viz', dest='viz', action='store_true',
        help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

def optical_flow(framepath):
    dataset_root = framepath
    output_root = join(framepath,'opticalflow')
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    dir_frames= join(dataset_root,'Frames')
    for j in range(0,749):
        im1 = np.array(Image.open(join(dir_frames,'{:d}'.format(j)+'.jpg')))
        im2 = np.array(Image.open(join(dir_frames,'{:d}'.format(j+1)+'.jpg')))
        im1 = cv2.resize(im1,(120,68))
        im2 = cv2.resize(im2,(120,68))
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.
    
        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
            
        u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
            
            
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    
        hsv = np.zeros(im1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(join(output_root,'{}'.format(j)+'.png'), rgb) 
