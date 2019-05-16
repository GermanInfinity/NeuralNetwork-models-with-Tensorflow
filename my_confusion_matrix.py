#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:11:07 2019

@author: Deeplizard and ugoslight
confusion matrix printerm

"""
import numpy as np
import matplotlib.pyplot as plt
import itertools


class my_confusion_matrix:
    def __init__(self, cm, classes):
        self.cm = cm
        self.classes = classes
        self.title = "Confusion Matrix"
        self.normalize = False
        self.cmap = plt.cm.Blues

    def plot(self):
    
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(self.cm, interpolation='nearest', cmap=self.cmap)
        plt.title(self.title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        
        if self.normalize:
            self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(self.cm)
    
        thresh = self.cm.max() / 2.
        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            plt.text(j, i, self.cm[i, j], horizontalalignment="center", color="white" if self.cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted label')