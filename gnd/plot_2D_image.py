#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:08:12 2018

@author: e57677

Analyze and simulate strain and misorientation images obtained from EBSD. 

inspired by work by Geraud Blatman
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL
from random import shuffle
from itertools import cycle
from matplotlib.mlab import griddata
from scipy.misc import imsave
from scipy.ndimage import measurements as ms

COLOR_CYCLE = cycle(["-b", "--k", "--g", "--r", ":m", "-c"])


class Image(object):
    """
    Image obtained either from measurements or from a simulation process
    """
    
    def __init__(self, data = None, filename = None, label=None, skiprows=0, data_type="data", \
                 size=(100, 100)):
        """
        Initialize the object from an image or ASCII filename
        """
#        self.filename, self.label = filename, label
        
        if (data_type == "file") & (filename is not None):
            x, y, z = np.loadtxt(filename, skiprows=skiprows, unpack=True)
            xlin = np.linspace(min(x), max(x), size[0])
            ylin = np.linspace(min(y), max(y), size[1])
            [xi, yi] = np.meshgrid(xlin, ylin)
            self.xi = xi
            self.yi = yi
            self.zi = griddata(x, y, z, xi, yi, 'linear')
        elif (data_type == "data") & (data is not None):
            x, y, z = data[:,0], data[:,1], data[:,2]
            xlin = np.linspace(min(x), max(x), size[0])
            ylin = np.linspace(min(y), max(y), size[1])
            [xi, yi] = np.meshgrid(xlin, ylin)
            self.xi = xi
            self.yi = yi
            self.zi = griddata(x, y, z, xi, yi, 'linear')
        else:
            raise ValueError("'data_type' should be 'data' or 'file'")
        
    def imshow(self, interpolation='bilinear', cmap=cm.RdYlGn,
                origin='lower' ):
        
        fig = plt.imshow(self.zi, interpolation=interpolation, cmap= cmap,
                origin='lower', extent=[self.xi.min(), self.xi.max(), self.yi.min(), self.yi.max()], 
                vmax=self.zi.max(), vmin=self.zi.min() )
        
        plt.axis('off')
        plt.colorbar()
        
        return fig
    
#    def get_connect_bin(self, threshold, plot=False):
#        """
#        Threshold the image and compute a connectivity measure of the resulting 
#        binarized image.
#        """
#        # Create binarized image
#        bin_image = self.image.copy()
#        bin_image[self.image > threshold] = 1
#        bin_image[self.image <= threshold] = 0
#        
#        # Compute the binarized image connectivity
#        lw, num = ms.label(bin_image) # detect and label the clusters
#        npix_array = np.bincount(lw.ravel())[1:] # count pixels in each cluster
#        connec = 1./(npix_array.sum()**2) * np.sum(npix_array**2)
#        
#        if plot:
#            plt.figure()
#            plt.imshow(self.image, cmap=cm.Greys_r)
#            plt.colorbar()
#            plt.title("Original image")
#            plt.figure()
#            plt.imshow(bin_image, cmap=cm.Greys_r)
#            
#            # Show clusters by areas
#            area = ms.sum(bin_image, lw, index=np.arange(lw.max() + 1))
#            areaImg = area[lw]
#            plt.figure()
#            im3 = plt.imshow(areaImg)
#            plt.title("Labelled clusters by area")
#            
#            # Draw bounding box
#            sliced = ms.find_objects(areaImg == areaImg.max())
#            if(len(sliced) > 0):
#                sliceX = sliced[0][1]
#                sliceY = sliced[0][0]
#                plotxlim=im3.axes.get_xlim()
#                plotylim=im3.axes.get_ylim()
#                plt.plot([sliceX.start, sliceX.start, sliceX.stop, sliceX.stop, \
#                          sliceX.start], \
#                         [sliceY.start, sliceY.stop, sliceY.stop, sliceY.start, \
#                          sliceY.start], color="red")
#                plt.xlim(plotxlim)
#                plt.ylim(plotylim)
#        
#        return connec
#        
#    
#    def plot_connect_curve(self, n_thresholds=75):
#        """
#        Plot the curve representing the connectivity of the image continuous 
#        field.
#        """
#        x = np.linspace(self.image.min(), self.image.max(), n_thresholds)
#        connec_list = []
#        for threshold in x:
#            connec_list += [ self.get_connect_bin(threshold) ]
#        fig = plt.figure()
#        plt.plot(x, connec_list, next(COLOR_CYCLE), label=self.label)
#        plt.xlabel(r"Threshold $t$")
#        plt.ylabel(r"Probability of connection $\Gamma(t)$")
#        plt.xlim(self.image.min(), self.image.max())
#        return fig
#        
#        
#    def cmp_images(self, others, n_thresholds=75):
#        """
#        Compare the connectivity curves of two images
#        """
#        fig = self.plot_connect_curve(n_thresholds)
#        for other in others:
#            connec_list = []
#            x = np.linspace(other.image.min(), other.image.max(), n_thresholds)
#            for threshold in x:
#                connec_list += [ other.get_connect_bin(threshold) ]
#            plt.plot(x, connec_list, next(COLOR_CYCLE), label=other.label)
#        plt.legend(loc=0)
        

if __name__ == "__main__":
    
#    import os
    import pandas as pd
    
    plt.close("all")
    
    filename = ("EBSD Data/1650-66 KAM 4th 15d reduced.txt")
    
    image_obj = Image(filename = filename, data_type="file", size=(5000,5000))
    
    fig = image_obj.imshow()
    
    fig = image_obj.imshow(cmap = cm.rainbow)
    
    fig.figure.savefig("figures/KAM.png" , bbox_inches = 'tight', dpi = 300)
    
    imsave("figures/KAM_1.png", image_obj.zi)
    
    

#    kam = pd.read_csv('EBSD Data/kam.txt', skiprows = 8, header = None, delimiter = ' ', usecols = [0, 1, 3])
#    kam.columns = ['x', 'y','KAM']
#    image_obj = Image(data = kam.values, data_type="data", size=(5000,5000))
#    imsave("figures/KAM_2.png", image_obj.zi)
    
    
