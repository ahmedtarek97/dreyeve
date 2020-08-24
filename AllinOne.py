# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:31:03 2020

@author: Abdelrahman Al-Wali
"""
from getFrames import get_frames
from Optical_Flow.demo import optical_flow
from seg import segment
from predication import predict
from visualization import visualize
from frameTovideo import generate_video
from os.path import join
import uuid
import os


def get_focus_of_attention(videopath, frames):
    path = videopath[:videopath.rfind('/')]
    mean_path = join(videopath[:videopath.rfind('/')],"mean_frame.png")

    outputfolder = join(path,"output")
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    print("///////////////////")
    print("getFrames Begin")
    print("///////////////////")
    get_frames(frames,videopath,join(outputfolder,'Frames'))
    print("///////////////////")
    print("OpticalFlow Begin")
    print("///////////////////")
    optical_flow(outputfolder,frames)
    print("///////////////////")
    print("Segmantention Begin")
    print("///////////////////")
    segment(outputfolder,frames)
    print("///////////////////")
    print("Prediction Begin")
    print("///////////////////")
    predict(outputfolder,mean_path,frames)
    print("///////////////////")
    print("Visualization Begin")
    print("///////////////////")
    visualize(outputfolder,frames)
    print("///////////////////")
    print("generateVideo Begin")
    print("///////////////////")
    generate_video(outputfolder,frames)
    
    
    
