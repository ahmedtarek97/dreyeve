# -*- coding: utf-8 -*-

from getFrames import get_frames
from Optical_Flow_Code.demo import optical_flow
from seg import segment
from predication import predict
from visualization import visualize
from frameTovideo import generate_video
from os.path import join
import uuid
import os


def get_focus_of_attention(videopath):
    id = uuid.uuid1()
    
    mean_path = join(videopath[:videopath.rfind('/')],"mean_frame.png")

    outputfolder = join("/output", id)
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
        
    get_frames(750,videopath,join(outputfolder,'Frames')).format()
    
    optical_flow(outputfolder)
    
    segment(outputfolder)
    
    predict(outputfolder,mean_path)
    
    visualize(outputfolder)
    
    generate_video(outputfolder)
    
    
    