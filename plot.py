#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:11:07 2019

@author: Deeplizard
plot code for classes of dataset

"""
import matplotlib.pyplot as plt 
import numpy as np

class plot:
    def __init__(self, image, title_label):
        self.ims = image
        self.figsize = (12, 6)
        self.rows = 1
        self.interp = False
        self.titles = title_label
    
    
    def plot_it(self):
        if type(self.ims[0]) is np.ndarray:
            self.ims = np.array(self.ims).astype(np.uint8)
            if (self.ims.shape[-1] != 3):
                self.ims = self.ims.transpose((0,2,3,1))
        f = plt.figure(figsize=self.figsize)
        
        self.cols = len(self.ims)// self.rows if len(self.ims) % 2 == 0 else len(self.ims)//self.rows + 1
        
        for i in range(len(self.ims)):
            sp = f.add_subplot(self.rows, self.cols, i+1)
            sp.axis('Off')
            if self.titles is not None:
                sp.set_title(self.titles[i], fontsize=16)
            plt.imshow(self.ims[i], interpolation=None if self.interp else 'none')