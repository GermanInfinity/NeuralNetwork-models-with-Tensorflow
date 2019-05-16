#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:56:35 2019

@author: ugoslight
Description: Using VGG16 Image model 
"""
import keras
import plot
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import my_confusion_matrix
import numpy as np 
#Keras has library of pretrained models
vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

#Model classifies a 1000 different categories 
#Must adopt model to classify our number of classes
type(vgg16_model) #Type model, not sequential. From Keras functional API 
#transform into sequential model 
model = Sequential()

for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
#take off last layer 
model.layers.pop()

#Have to a sequential model

for layer in model.layers:
    layer.trainable = False
    
model.add(Dense(2, activation='softmax'))
model.summary()



model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=6, validation_data=valid_batches, validation_steps=2, epochs=5, verbose=2)

test_imgs, test_labels = next(test_batches)
my_plot = plot.plot(test_imgs, title_label=test_labels)
#my_plot.plot_it()

#Look at test batches
test_labels = test_labels[:, 0]
test_labels
#Predicting with our model
predictions = model.predict_generator(test_batches, steps=1, verbose=0) #steps is number of steps to grab all data
cm = confusion_matrix(test_labels, np.round(predictions[:,0]))

cm_plot_labels = ['person', 'no_person']
classes=['person','noperson']
print_cm = my_confusion_matrix.my_confusion_matrix(cm, classes)
print_cm.plot()