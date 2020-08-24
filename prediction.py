#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow
import sys 
sys.path.append("Model/")
import os
os.environ['KERAS_BACKEND'] ="tensorflow"
import keras.backend as K
from tensorflow.python import keras
K.image_data_format()
from keras.models import Model
from keras.layers import Input, Reshape, merge, Lambda, Activation, LeakyReLU,concatenate,add
from keras.layers import Convolution3D, MaxPooling3D, Convolution2D,UpSampling2D
from keras.utils.data_utils import get_file
from train.config import simo_mode
from computer_vision_utils.io_helper import read_image, normalize
from computer_vision_utils.tensor_manipulation import resize_tensor
from computer_vision_utils.stitching import stitch_together
from train.utils import seg_to_colormap
from metrics.metrics import kld_numeric, cc_numeric
import seaborn as sb 
import matplotlib.pyplot as pl
import numpy as np
from tqdm import tqdm
from os.path import join
import cv2
from train.models import SaliencyBranch , DreyeveNet 



def load_dreyeve_sample(sequence_dir, sample, mean_dreyeve_image, frames_per_seq=16, h=448, w=448, ):
    """
    Function to load a dreyeve_sample.

    :param sequence_dir: string, sequence directory (e.g. 'Z:/DATA/04/').
    :param sample: int, sample to load in (15, 7499). N.B. this is the sample where prediction occurs!
    :param mean_dreyeve_image: mean dreyeve image, subtracted to each frame.
    :param frames_per_seq: number of temporal frames for each sample
    :param h: h
    :param w: w
    :return: a dreyeve_sample like I, OF, SEG
    """

    h_c = h_s = h // 4
    w_c = w_s = h // 4

    I_ff = np.zeros(shape=(1, 1, h, w,3), dtype='float32')
    I_s = np.zeros(shape=(1, frames_per_seq, h_s, w_s,3), dtype='float32')
    I_c = np.zeros(shape=(1, frames_per_seq, h_c, w_c,3), dtype='float32')
    SEG_ff = np.zeros(shape=(1, 1, h, w,3), dtype='float32')
    SEG_s = np.zeros(shape=(1,frames_per_seq, h_s, w_s,3), dtype='float32')
    SEG_c = np.zeros(shape=(1, frames_per_seq, h_c, w_c,3), dtype='float32')
    OF_ff = np.zeros(shape=(1,  1, h, w,3), dtype='float32')
    OF_s = np.zeros(shape=(1,  frames_per_seq, h_s, w_s,3), dtype='float32')
    OF_c = np.zeros(shape=(1,  frames_per_seq, h_c, w_c,3), dtype='float32')
    
   
    for fr in range(0, frames_per_seq):
        offset = sample - frames_per_seq + 1 + fr   # tricky
        
        
        # read image
        x = read_image(join(sequence_dir,'Frames','{}.jpg'.format(offset)),
                       channels_first=False, resize_dim=(h, w)) - mean_dreyeve_image
        I_s[0, fr, :, :,:] = resize_tensor(x,new_shape=(h_s, w_s))
        
        
        # read semseg
        seg = resize_tensor(np.load(join(sequence_dir,'Seg', '{}.npz'.format(offset)))['arr_0'],
                            new_shape=(h, w))
        SEG_s[0,fr,:, :, :] = resize_tensor(seg, new_shape=(h_s, w_s))
        
        # read of
        
        of = read_image(join(sequence_dir,'Optical', '{}.png'.format(offset + 1)),
                        channels_first=False, resize_dim=(h, w))
        of -= np.mean(of, axis=(1, 2), keepdims=True)  # remove mean
        OF_s[0, fr, :, :, :] = resize_tensor(of, new_shape=(h_s, w_s))
        

    I_ff[0,0, :, :,:] = x
    SEG_ff[0,0, :, :, :] = seg
    OF_ff[0, 0, :, :, :] = of

    return [I_ff, I_s,I_c,OF_ff,OF_s,OF_c,SEG_ff, SEG_s, SEG_c]


def predict(filepath,mean_path):
    
    frames_per_seq, h, w = 16, 448, 448
    dreyevenet_model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    dreyevenet_model.load_weights('Best_Weights/Tune.h5')

    mean_dreyeve_image = read_image(mean_path,channels_first=False, resize_dim=(h, w))
    output_dir = join(filepath,'npz')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for sample in tqdm(range(0 + 15 , 749)):

        X = load_dreyeve_sample(filepath, sample=sample, mean_dreyeve_image=mean_dreyeve_image,
                                frames_per_seq=frames_per_seq, h=h, w=w)#heeeerrrrrr
        #Y_dreyevenet = dreyevenet_model.predict(X)[0]  # get only [fine_out][remove batch]
        Y_image = dreyevenet_model.predict(X)[0]  # predict on image
        #output folder  here output will be npz files not photos
        np.savez_compressed(output_dir + "/{}".format(sample),Y_image)
    
    
    