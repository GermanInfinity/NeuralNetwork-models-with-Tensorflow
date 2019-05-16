#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:36:09 2019

@author: ugoslight
"""
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Reshape, Permute, Conv2D, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU 
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical 
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt 
from keras.utils.vis_utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import plot
import my_confusion_matrix
#Dataset
import h5py
import tensorflow as tf

#train_path = '/Users/ugoslight/Desktop/lab/train'
train_path = '/Users/ugoslight/Downloads/class/train'
valid_path = '/Users/ugoslight/Downloads/class/valid'
test_path = '/Users/ugoslight/Downloads/class/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['person','noperson'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),classes=['person','noperson'], batch_size=5)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),classes=['person','noperson'], batch_size=10)

#Sequential model. 
#1st layer - conv 2D image layer, 32: num of output filters in convolution
#3,3 kernel size: specifies width and height of 2D convolution window 
#224,224,3: height width and channel of our images
#Flatener, flattens output to 1D tensor
#1D tensor fed into Dense layer with two nodes that classify output 
model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)), Flatten(), Dense(2, activation='softmax'),])
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#total number of images before declaring epoch is done 
model.fit_generator(train_batches, steps_per_epoch=3, validation_data=valid_batches,
                    validation_steps=2, epochs=5, verbose=2)

test_imgs, test_labels = next(test_batches)
my_plot = plot.plot(test_imgs, title_label=test_labels)
#my_plot.plot_it()

#Look at test batches
test_labels = test_labels[:, 0]


#Predicting with our model
predictions = model.predict_generator(test_batches, steps=1, verbose=0) #steps is number of steps to grab all data
cm = confusion_matrix(test_labels, predictions[:,0])

cm_plot_labels = ['person', 'no_person']
classes=['person','noperson']
print_cm = my_confusion_matrix.my_confusion_matrix(cm, classes)
print_cm.plot()
#Predicted person 7 times correctrly and 3 times wrongly. 
#Never predicted noperson
#Import new model and then fine tune.