#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:40:47 2018

@author: chumai

data source
https://data.worldbank.org/country/united-states
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

data = pd.read_excel('../data/world_population.xls',sheetname = 'Data', skiprows = 3)
data.info()
data.head()
data.columns

data['Indicator Name'].unique()
data['Indicator Code'].unique()

#%%

data.drop([u'Country Code', u'Indicator Name', u'Indicator Code'], axis = 1, inplace = True)

data.set_index('Country Name', inplace = True)

#%%
df = data.sum(axis = 0, numeric_only = True)
data.loc['Total',df.index.tolist()] = df.values

#%%
data.tail()

data.loc[data.index.str.contains('United')]
#%%
plt.figure()
(1.0e-6 * data.loc['United States',:] ).plot()
plt.xlabel('United States')
#%%
plt.figure()
(1.0e-6 * data.loc['United Arab Emirates',:] ).plot()
#%%
plt.figure()
(1.0e-6 * data.loc[data.index.str.contains('United'),:].transpose() ).plot(logy= True)
#plt.xlabel('United States')
#%%
data.loc[data.index.str.contains('Euro')]
plt.figure()
(1.0e-6 * data.loc['European Union',:] ).plot()