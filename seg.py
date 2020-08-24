#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:37:26 2020

@author: a_alwali96
"""
import six
import sys
sys.path.append('LightWeight Semantic Segmentation/')
from models.resnet import rf_lw152
from utils.helpers import prepare_img
import cv2
import numpy as np
import torch
import os
from os.path import join

def segment(framepath,frames):
    cmap = np.load('LightWeight Semantic Segmentation/utils/cmap.npy')
    has_cuda = torch.cuda.is_available()
    n_classes = 60
    
    # Initialise models
    model_inits = { 
    'rf_lw152_context'   : rf_lw152,
    }

    models = dict()
    for key,fun in six.iteritems(model_inits):
        net = fun(n_classes, pretrained=True).eval()
        if has_cuda:
            net = net.cuda()
        models[key] = net
        
    dataset_root = framepath
    output_root = join(framepath,'Seg')
    if not os.path.exists(output_root):
        os.mkdir(output_root)
        
    
    with torch.no_grad():
        dir_frames= join(dataset_root,'Frames')
        dir_out_OF= output_root
        if not os.path.exists(dir_out_OF):
            os.mkdir(dir_out_OF)
        for j in range(0,frames):
            img = np.array(cv2.imread(join(dir_frames,'{:d}'.format(j)+'.jpg'))[:, :, ::-1])
            orig_size = img.shape[:2][::-1]
            
            img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
            if has_cuda:
                img_inp = img_inp.cuda() 
            for mname, mnet in six.iteritems(models):
                segm = mnet.cuda()(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
                segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
                segm = cmap[segm.argmax(axis=2).astype(np.uint8)]
		
            print(j)
            np.savez_compressed(join(dir_out_OF,'{}'.format(j)+'.npz'), segm)
