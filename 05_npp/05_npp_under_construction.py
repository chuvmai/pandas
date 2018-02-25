#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:06:12 2018

@author: chumai
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

#%%

df = pd.read_excel('../data/NPP_under_construction.xls')
df = pd.read_excel('../data/NPP_in_operation.xls')

df.info()

df.head(10)

df.tail(15)

df.drop(range(60,71), axis = 0, inplace = True)

df['Reactor Type'].unique()

df['Reactor Type'].value_counts()

# pie number of each reactor type
plt.figure()
df['Reactor Type'].value_counts().plot.pie()

# pie MWe 
# sum of MWe
plt.figure()
df.groupby('Reactor Type')['Total MWe'].sum().plot.pie()
plt.ylabel('Total MWe')

# mean of MWe
plt.figure()
df.groupby('Reactor Type')['Total MWe'].mean().plot.pie()
plt.ylabel('Mean MWe')


df.groupby('Reactor Type')['Total MWe'].describe().sort_values(by = 'count', ascending = False)

# where a specific type of reactor is built
df[df['Reactor Type'] == 'ABWR']

for re in df['Reactor Type'].unique().tolist():
    print df[df['Reactor Type'] == re]
