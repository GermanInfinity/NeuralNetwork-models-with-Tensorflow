#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:16:53 2019

@author: ugoslight
Description: CRNN build
"""

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense Activation 
from keras.layer import Reshape, Lambda, BatchNormalization 
from keras.layers.merge import add, concatenate
from keras.models import Model 
from keras.layers.recurrent import LSTM 
from parameter import *
K.set_learning_phase(0)


##Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred - y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_Model(training):
    input_shape = (img_w, img_h, 1) # 128, 64, 1
    
    #Make Network
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    
    #Convolutional layers
    inner = Conv2D(64)