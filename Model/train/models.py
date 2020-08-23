
import tensorflow
print(tensorflow.__version__)

import sys 
sys.path.append("/content/drive/My Drive/dreyeve project/experiments")

import os
os.environ['KERAS_BACKEND'] ="tensorflow"

import keras.backend as K

K.set_image_data_format("channels_last")
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



def resize_tensor(tensor, new_shape):
    """
    Resize a numeric input 3D tensor with opencv. Each channel is resized independently from the others.
    
    Parameters
    ----------
    tensor: ndarray
        Numeric 3D tensor of shape (channels, h, w)
    new_shape: tuple
        Tuple (new_h, new_w)

    Returns
    -------
    new_tensor: ndarray
        Resized tensor having size (channels, new_h, new_w)
    """
    channels = tensor.shape[2]
    new_tensor = np.zeros(shape=(channels,) + new_shape)
    new_tensor=new_tensor.T
    for i in range(0, channels):
        new_tensor[:,:,i] = cv2.resize(tensor[:,:,i],dsize=new_shape)

    return new_tensor

def CoarseSaliencyModel(input_shape, pretrained, branch=''):
    """
    Function for constructing a CoarseSaliencyModel network, based on C3D.
    Used for coarse prediction in SaliencyBranch.
    :param input_shape: in the form (channels, frames, h, w).
    :param pretrained: Whether to initialize with weights pretrained on action recognition.
    :param branch: Name of the saliency branch (e.g. 'image' or 'optical_flow').
    :return: a Keras model.
    """
    fr, h, w,c = input_shape
    assert h % 8 == 0 and w % 8 == 0, 'I think input shape should be divisible by 8. Should it?'
    

    # input_layers
    model_in = Input(shape=input_shape, name='input')
    # encoding net
    H = Convolution3D(64, (3, 3, 3), activation='relu', padding='same', name='conv1', strides=(1, 1, 1))(model_in)
    H = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(H)
    H = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(H)
    H = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(H)
    H = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),padding='valid', name='pool3')(H)
    H = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(H)
    H = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(4, 1, 1), strides=(4, 1, 1),padding='valid', name='pool4')(H)
    
    
    H = Reshape(target_shape=(h // 8, w // 8,512))(H)  # squeeze out temporal dimension

    model_out = UpSampling2D(size=(2,2),interpolation='bilinear',name='{}_81x_upsampling'.format(branch))(H)
    model_out = UpSampling2D(size=(2,2),interpolation='bilinear',name='{}_82x_upsampling'.format(branch))(model_out)
    model_out = UpSampling2D(size=(2,2),interpolation='bilinear',name='{}_83x_upsampling'.format(branch))(model_out)
    
    model = Model(inputs=model_in, outputs=model_out, name='{}_coarse_model'.format("image"))

   

    return model

def SaliencyBranch(input_shape, c3d_pretrained, branch=''):
    """
    Function for constructing a saliency model (coarse + fine). This will be a single branch
    of the final DreyeveNet.

    :param input_shape: in the form (channels, frames, h, w). h and w refer to the fullframe size.
    :param branch: Name of the saliency branch (e.g. 'image' or 'optical_flow').
    :return: a Keras model.
    """
    fr, h, w,c = input_shape
    

    coarse_predictor = CoarseSaliencyModel(input_shape=(fr, h // 4, w // 4,c), pretrained=c3d_pretrained, branch=branch)

    ff_in = Input(shape=(1, h, w,c), name='{}_input_ff'.format(branch))
    small_in = Input(shape=(fr, h // 4, w // 4,c), name='{}_input_small'.format(branch))
    crop_in = Input(shape=(fr, h // 4, w // 4,c), name='{}_input_crop'.format(branch))

    # coarse + refinement
    ff_last_frame = Reshape(target_shape=(h, w,c))(ff_in)  # remove singleton dimension
    coarse_h = coarse_predictor(small_in)
    coarse_h = Convolution2D(1, (3, 3), padding='same', activation='relu')(coarse_h)
    coarse_h = UpSampling2D(size=(2,2),interpolation='bilinear', name='{}_21x_upsampling'.format(branch))(coarse_h)
    coarse_h = UpSampling2D(size=(2,2),interpolation='bilinear', name='{}_22x_upsampling'.format(branch))(coarse_h)
   
    
    fine_h = concatenate([coarse_h, ff_last_frame], axis=3 , name='{}_full_frame_concat'.format(branch))
    fine_h = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='{}_refine_conv1'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='{}_refine_conv2'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(8, (3, 3), padding='same', kernel_initializer='he_normal', name='{}_refine_conv3'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(1, (3, 3), padding='same', kernel_initializer='glorot_uniform', name='{}_refine_conv4'.format(branch))(fine_h)
    fine_out = Activation('relu')(fine_h)

    if simo_mode:
       
        fine_out = Lambda(lambda x: K.repeat_elements(x, rep=2, axis=1),
                          output_shape=(2, h, w), name='prediction_fine')(fine_out)
    else:
        fine_out = Activation('linear', name='prediction_fine')(fine_out)

    # coarse on crop
    
    crop_h = coarse_predictor(crop_in)
    crop_h = Convolution2D(1, (3, 3), padding='same', kernel_initializer='glorot_uniform', name='{}_crop_final_conv'.format(branch))(crop_h)
    crop_out = Activation('relu', name='prediction_crop')(crop_h)

    model = Model(inputs=[ff_in,small_in,crop_in], outputs=[fine_out, crop_out],
                  name='{}_saliency_branch'.format(branch))

    return model
def DreyeveNet(frames_per_seq, h, w):
    """
    Function for constructing the whole DreyeveNet.

    :param frames_per_seq: how many frames in each sequence.
    :param h: h (fullframe).
    :param w: w (fullframe).
    :return: a Keras model.
    """
    # get saliency branches
    im_net = SaliencyBranch(input_shape=(frames_per_seq, h, w,3), c3d_pretrained=True, branch='image')
    im_net.load_weights('/home/a_alwali96/GP-2020/Weights/image.h5')
    of_net = SaliencyBranch(input_shape=(frames_per_seq, h, w,3), c3d_pretrained=True, branch='optical_flow')
    of_net.load_weights('/home/a_alwali96/GP-2020/Weights/flow.h5')
    seg_net = SaliencyBranch(input_shape=(frames_per_seq, h, w,3), c3d_pretrained=False, branch='segmentation')
    seg_net.load_weights('/home/a_alwali96/GP-2020/Weights/segm.h5')

    # define inputs
    X_ff = Input(shape=(1, h, w, 3), name='image_fullframe')
    X_small = Input(shape=( frames_per_seq, h // 4, w // 4,3), name='image_resized')
    X_crop = Input(shape=(frames_per_seq, h // 4, w // 4,3), name='image_cropped')

    OF_ff = Input(shape=( 1, h, w,3), name='flow_fullframe')
    OF_small = Input(shape=(frames_per_seq, h // 4, w // 4,3), name='flow_resized')
    OF_crop = Input(shape=(frames_per_seq, h // 4, w // 4,3), name='flow_cropped')

    SEG_ff = Input(shape=(1, h, w,3), name='semseg_fullframe')
    SEG_small = Input(shape=( frames_per_seq, h // 4, w // 4,3), name='semseg_resized')
    SEG_crop = Input(shape=(frames_per_seq, h // 4, w // 4,3), name='semseg_cropped')

    x_pred_fine, x_pred_crop = im_net([X_ff, X_small, X_crop])
    of_pred_fine, of_pred_crop = of_net([OF_ff, OF_small, OF_crop])
    seg_pred_fine, seg_pred_crop = seg_net([SEG_ff, SEG_small, SEG_crop])
    fine_out = add([x_pred_fine, of_pred_fine, seg_pred_fine],name='merge_fine_prediction')
    fine_out = Activation('relu', name='prediction_fine')(fine_out)

    crop_out = add([x_pred_crop, of_pred_crop, seg_pred_crop],name='merge_crop_prediction')
    crop_out = Activation('relu', name='prediction_crop')(crop_out)

    model = Model(inputs =[X_ff, X_small, X_crop, OF_ff, OF_small, OF_crop, SEG_ff, SEG_small, SEG_crop],
                  outputs =[fine_out, crop_out], name='DreyeveNet')

    return model
# tester function
if __name__ == '__main__':
    model = SaliencyBranch(input_shape=( 16, 448, 448,3), c3d_pretrained=True, branch='image')
    model.summary()

