#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#import cv2

def get_frames(number_of_frames,v_path,im_path):
    cap= cv2.VideoCapture(v_path)
    i=0
    counter=0
    while(cap.isOpened()):
    
        ret, frame = cap.read()
        if ret == False:
            break
        
        
        cv2.imwrite(im_path+str(i)+'.jpg',frame)
        i+=1
        counter+=1
   
        if(counter == number_of_frames):
            break
        print(counter)
    cap.release()
    cv2.destroyAllWindows()
   