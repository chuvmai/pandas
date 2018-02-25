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


import datetime as dt
#%%

df = pd.read_excel('../data/NPP_in_operation.xls')

df.info()

df.head(10)

df.tail(15)

df.drop(df.index[350:], axis = 0, inplace = True)


df['Date Connected'].astype('int64')

df['Date Connected'] = pd.to_datetime(df['Date Connected'].astype('int64'), format ='%Y')

df['Year connected'] = df['Date Connected'].dt.year

df['Age'] = dt.datetime.today().year - df['Date Connected'].dt.year
#%%

df['Reactor Type'].unique()

df['Reactor Type'].value_counts().sort_values(ascending = False)

# pie number of each reactor type
plt.figure()
df['Reactor Type'].value_counts().plot.pie()

# pie MWe 
# sum of MWe
plt.figure()
df.groupby('Reactor Type')['Net Capacity (MW)'].sum().plot.pie()
plt.ylabel('Net Capacity (MW)')

# mean of MWe
plt.figure()
df.groupby('Reactor Type')['Net Capacity (MW)'].mean().plot.pie()
plt.ylabel('Mean MWe')


df.groupby('Reactor Type')['Net Capacity (MW)'].describe().sort_values(by = 'count', ascending = False)

# where a specific type of reactor is built
df[df['Reactor Type'] == 'ABWR']

for re in df['Reactor Type'].unique().tolist():
    print df[df['Reactor Type'] == re]

df.sort_values(by = 'Age', ascending = False).head(20)[[u'Country',u'Reactor Type',u'Net Capacity (MW)','Age']]

df2 = df.loc[ df['Country'] == 'France']

df2.describe()

df2.head()


df2['Net Capacity (MW)'].sum()

df2.sort_values(by = 'Age', ascending = False).head(20)
# show Bugey picture

plt.figure()
df2['Age'].plot.hist(bins = 10)

plt.figure()
df2['Age'].plot.kde()