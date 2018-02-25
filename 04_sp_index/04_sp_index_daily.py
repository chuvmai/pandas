#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:20:42 2018

@author: chumai

datetime format
https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior

"""

import pandas as pd
import matplotlib.pyplot as plt
rcParams = { 'axes.grid': False,
             'axes.labelsize': 14,
             'xtick.labelsize': 10,
             'ytick.labelsize': 10,
             'font.size': 14,
             'legend.fontsize': 12.0,
             'lines.linewidth': 5.0,
             'figure.figsize': (8.0,6.0)}
plt.rcParams.update(rcParams)

import datetime as dt

import scipy.stats as stats

data = pd.read_csv('../data/S&P 500 Daily Data.csv')
data.info()
data.head()
data.tail()

data.drop(  data.iloc[-2:,:].index , axis = 0, inplace = True)


#%%

data['Date'] = pd.to_datetime(data['Date'], format= '%b %d, %Y')

data.set_index('Date', inplace = True)

data.drop('Vol.', axis= 1, inplace = True)

for col in ['Price', 'Open', 'High', 'Low' ]:
    data[ col ] = data[ col ].str.replace(",","").astype('float64')
    
data.index.rename('Time',inplace = True)
data.sort_index(inplace = True)