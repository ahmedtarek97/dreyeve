#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:43:55 2020

@author: a_alwali96
"""

import os 
import cv2  
from os.path import join

def generate_video(folderpath,frames):
        image_folder =  join(folderpath,'jpg')
        video_name = join(folderpath,'predict.avi')

        frame = cv2.imread(os.path.join(image_folder, '15.jpg')) 
  
        # setting the frame width, height width 
        # the width, height of first image 
        height, width, layers = frame.shape   
  
        video = cv2.VideoWriter(video_name, 0, 25, (width, height))  
  
        # Appending the images to the video one by one 
        for image in range(15,frames):  
            video.write(cv2.imread(os.path.join(image_folder, str(image) + '.jpg')))  
      
        # Deallocating memories taken for window creation 
        video.release()  # releasing the video generated 
        
  
