#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os 
import cv2  
from os.path import join

def generate_video(folderpath):
    for i in range(74 , 75):
        image_folder =  join(folderpath,'jpg')
        video_name = join(folderpath,'predict.avi')

        frame = cv2.imread(os.path.join(image_folder, '15.jpg')) 
  
        # setting the frame width, height width 
        # the width, height of first image 
        height, width, layers = frame.shape   
  
        video = cv2.VideoWriter(video_name, 0, 25, (width, height))  
  
        # Appending the images to the video one by one 
        for image in range(15,749):  
            video.write(cv2.imread(os.path.join(image_folder, str(image) + '.jpg')))  
      
        # Deallocating memories taken for window creation
        cv2.destroyAllWindows()  
        video.release()  # releasing the video generated 
        
  
