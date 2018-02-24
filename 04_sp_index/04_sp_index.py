#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:15:51 2018

@author: chumai

data source

https://www.investing.com/indices/us-spx-500-historical-data

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

import scipy as sp
import scipy.stats as stats
#%%

data = pd.read_csv('../data/S&P 500 Monthly Data.csv')
data.info()
data.head()
data.tail()

data.drop(  data.iloc[-2:,:].index , axis = 0, inplace = True)


#%% fastest way convert columns of string to date time
data['Date'] = pd.to_datetime(data['Date'], format= '%b %y')
#%%
dt.datetime.strptime('Feb 18','%b %y')
#apply this function to the column
data['Date'] = data['Date'].apply(lambda x: dt.datetime.strptime(x,'%b %y'))
#%% vice-versa, convert datetime column to strings
data['Date'].dt.strftime('%y %b')
data['Date'].dt.strftime('%Y %b')

#%%
data.drop('Vol.', axis = 1, inplace = True)
#%%
for col in [u'Price', u'Open', u'High', u'Low']:
    data[col]  = data[col].str.replace(",","").astype('float64')
#%%
data.set_index('Date', inplace = True)

data.index.rename('Time', inplace = True)
data.sort_index(ascending = True, inplace = True)

#%% correlation between quantities
data.corr()
#%%
plt.figure()
data.plot()

#%%
plt.figure()
data['Change %'].plot()

#%%
plt.figure()
data.loc[(data.index >= '1995-02-01') & (data.index <= '2018-02-01'),'Change %'].plot()
#%%
plt.figure()
data['Change %'].plot.hist(bins = 40)
#%% 
plt.figure()
data.loc[(data.index >= '1995-02-01') & (data.index <= '2018-02-01'),'Change %'].plot.hist()
#%%
stats.shapiro(data.loc[:,'Change %'])

#%%
stats.kstest(data.loc[:,'Change %'],'norm')
#%%
stats.normaltest(data.loc[:,'Change %'])
#%%
(test_statistic,p_value) = stats.shapiro(data.loc[(data.index >= '1995-02-01') & (data.index <= '2018-02-01'),'Change %'])

#%%
plt.figure()
data[ [u'Price', u'Open', u'High', u'Low']].plot()
#%%
for col in [u'Price', u'Open', u'High', u'Low']:
    plt.figure()
    data[ col].plot()
    plt.ylabel(col)
#%%
plt.figure()
data[ [u'Price', u'Open', u'High', u'Low']].plot()

#%%
plt.figure()
data.loc[data.index.isin(pd.date_range('2005-02-01','2018-02-01')), [u'Price', u'Open', u'High', u'Low']].plot()
plt.xlabel('')

#%%
plt.figure()
data.loc[(data.index >= '1995-02-01') & (data.index <= '2018-02-01'), [u'Price', u'Open', u'High', u'Low']].plot()
#%%
plt.figure()
data.loc[data.index.isin(pd.date_range('2005-02-01','2018-02-01')), [u'Price']].plot()
plt.xlabel('')

#%%
#%%
plt.figure()
data[ [u'Price', u'Change %']].plot()
#%%
plt.figure()
data.loc[ data.index.isin(pd.date_range('2010-02-01','2018-02-01')), [u'Price', u'Change %']].plot()