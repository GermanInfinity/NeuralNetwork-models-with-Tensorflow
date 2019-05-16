#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:17:00 2019

@author: ugoslight

DESCRIPTION

A Feedforward Neural Network
Input --> Weight --> Hidden layer 1 (activation function) --> Weights --> 
Hidden layer 2 (activation function) --> Weights --> Output layer

pass data straight through,  
at the end we compare output to intended output. 
How close is it? Compare with loss function 

Compare output to intended output --> Cost function (i.e Cross Entropy)
Optimization function (optimizer) --> Minimize cost (AdamOptimizer, SGD, etc.)

Opt func: Goes backward and manipulates weights. <-- Backpropagation

Feed forward + Backprop = epoch, done maybe 10, 15, 20 times
Each time lowering cost function, till it levels out. (Diminishing returns)

One element is hot or on. 
"""

import tensforflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')