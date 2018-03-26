#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:23:04 2018

@author: e57677
"""
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imsave

from plot_2D_image import Image

import copy as cp
import time

import numba

import pickle
#%%
gnd = pd.read_csv('EBSD Data/GND.txt', skiprows = 8, header = None, delimiter = ' ', usecols = [0, 1, 3])
gnd.columns = ['x', 'y','GND']
gnd.head(10)


kam = pd.read_csv('EBSD Data/kam.txt', skiprows = 8, header = None, delimiter = ' ', usecols = [0, 1, 3])
kam.columns = ['x', 'y','KAM']
kam.head(10)


iq = pd.read_csv('EBSD Data/IQ.txt', skiprows = 8, header = None, delimiter = ' ', usecols = [0, 1, 2])
iq.columns = ['x', 'y','IQ']
iq.head(20)


grain = pd.read_csv('EBSD Data/grain_file_1.txt', skiprows = 15, header = None, delimiter = '\s+')
grain.columns = ['phi1', 'PHI', 'phi2', 'x', 'y', 'IQ', 'CI', 'Fit', 'Grain ID', 'edge', 'phase']
grain.head(10)

#%%
#grain id = 0, this pixel is a joint

grain['Join'] =  0

grain.loc[ grain['Grain ID'] == 0, 'Join'] = 1
grain.head(20)
#%%
image_obj = Image(data = gnd.values, data_type="data", size=(1000,1000))

fig = image_obj.imshow(cmap = cm.jet)

fig.figure.savefig("figures/gnd.png" , bbox_inches = 'tight', dpi = 500)


#%%
image_obj = Image(data = kam.values, data_type="data", size=(1000,1000))

fig = image_obj.imshow(cmap = cm.jet)

fig.figure.savefig("figures/kam.png" , bbox_inches = 'tight', dpi = 500)
#%%

image_obj = Image(data = grain[['x', 'y', 'CI']].values, data_type="data", size=(100,100))

fig = image_obj.imshow()

fig = image_obj.imshow(cmap = cm.rainbow)

fig = image_obj.imshow(cmap = cm.jet)

fig.figure.savefig("figures/CI.png" , bbox_inches = 'tight', dpi = 300)

imsave("figures/CI2.png", image_obj.zi)

#%% dataframe of join grains

joingrain = grain.loc[ grain['Join'] == 1, ['x', 'y'] ]

joingrain.head()

grain[['x','y']].head()

grain.loc[0,['x','y']]

joingrain['x'] - 1

joingrain - [1,2]

(joingrain - [1,2]).head()


#%%
start_time = time.time()

dist = cp.deepcopy(joingrain)

pixel = [6.3,0.0]
# calculate distance
dist['dist'] = np.sqrt( np.square((dist - pixel)).sum(axis = 1) )

dist.head()


dist.loc[ dist['dist'].idxmin() ]

dist.loc[ dist['dist'].idxmin() ][['x', 'y']]
dist['dist'].min()

print("--- %s seconds ---" % (time.time() - start_time))

#%%

def min_dist( pixel, joingrain = joingrain):
    # function to find min distance from a pixel to all join grains
    

    return np.sqrt( np.square((joingrain - pixel)).sum(axis = 1) ).min()
    
#%% calculate minimal distance to join grains 

grain.head()

start_time = time.time()

grain['dist'] = grain[['x','y']].apply(lambda x: min_dist(x[['x','y']], joingrain = joingrain), axis=1)

print("--- %s seconds ---" % (time.time() - start_time))


#%% calculate minimal distance to join grains USING NUMBA
@numba.jit
def min_dist( pixel, joingrain = joingrain):
    # function to find min distance from a pixel to all join grains
    

    return np.sqrt( np.square((joingrain - pixel)).sum(axis = 1) ).min()
#%%

grain.head()

start_time = time.time()

grain['dist'] = grain[['x','y']].apply(lambda x: min_dist(x[['x','y']], joingrain = joingrain), axis=1)

print("--- %s seconds ---" % (time.time() - start_time))

#%%
grain.corr()

#%%
pickle.dump(grain,open('data/grain.p','w'))

#%%
grain = pickle.load( open('data/grain.p') )

#%%

df = pd.concat([grain.set_index(['x','y']), gnd.set_index(['x','y']) ], axis = 1, join = 'inner').reset_index()

df.corr()

#plt.matshow( df.corr())

#%%
plt.figure(figsize = (16.0,12.0))
sns.heatmap(df.corr(),xticklabels= df.corr().columns.values,yticklabels=df.corr().columns.values, cmap = cm.PiYG, center = 0., annot = True, fmt = '.2f' )
