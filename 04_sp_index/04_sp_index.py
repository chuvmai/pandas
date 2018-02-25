#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:15:51 2018

@author: chumai

data source

https://www.investing.com/indices/us-spx-500-historical-data

datetime format
https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior

datetime resampling aliases
http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
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
import scipy.stats as stats


import datetime as dt
import matplotlib.dates as mdates
#%%

data = pd.read_csv('../data/S&P 500 Monthly Data.csv')

data.info()
# non-null object
data.head()
# not float but string with , Vol. is 0.0 for all
data.tail()
# remove two extra rows
data.drop(  data.iloc[-2:,:].index , axis = 0, inplace = True)


#%% fastest way convert columns of string to date time
data['Date'] = pd.to_datetime(data['Date'], format= '%b %y')
data.head()
#%% the long version using datetime package
dt.datetime.strptime('Feb 18','%b %y')
#apply this function to the column
data['Date'] = data['Date'].apply(lambda x: dt.datetime.strptime(x,'%b %y'))
#%% vice-versa, convert datetime column to strings
data['Date'].dt.strftime('%y %b')
data['Date'].dt.strftime('%Y %b')

#%% drop Vol. columns
data.drop('Vol.', axis = 1, inplace = True)
#%% replace , in columnns with nothing, convert to float64
for col in [u'Price', u'Open', u'High', u'Low']:
    data[col]  = data[col].str.replace(",","").astype('float64')
#%% set Date column as index, change its name, sort by ascending order
data.set_index('Date', inplace = True)

data.index.rename('Time', inplace = True)
data.sort_index(ascending = True, inplace = True)

#%% correlation between quantities
data.corr()

#%%

data.describe()

# high CoV -> very volatile
data.describe().loc['std'] / data.describe().loc['mean']

data['Change %'].describe()

data['Change %'].mean()

data['Change %'].std()

#%% sort by Change % then plot 

# the worst months
data.sort_values(by = ['Change %'], axis = 0).head(10)

# nice xlabel
plt.figure()
data['Change %'].sort_values(ascending = True, axis = 0).head(10).plot(linestyle = '', marker = 'o', markersize = 5.)
plt.ylabel('Change %')

plt.figure()
data['Change %'].sort_values(ascending = False, axis = 0).head(10).plot(linestyle = '', marker = 'o', markersize = 5.)
plt.ylabel('Change %')


'''
not nice bar plot, to work on if possible
'''
plt.figure()
data.sort_values(by = ['Change %'], axis = 0).head(10).plot.bar()

plt.figure()
data['Change %'].sort_values(ascending = False, axis = 0).head(10).plot.bar()
plt.ylabel('Change %')

# not working
fig, ax = plt.subplots()
data['Change %'].sort_values(ascending = False, axis = 0).head(10).plot.bar()
ax.xaxis.set_major_locator(mdates.MonthLocator())
#ax.xaxis.set_minor_locator(mdates.MonthLocator())
monthFmt = mdates.DateFormatter('%Y %b')
ax.xaxis.set_major_formatter(monthFmt)

plt.figure()
abs(data['Change %']).sort_values(ascending = True, axis = 0).head(10).plot.bar()

plt.figure()
data['Change %'].reindex( data['Change %'].abs().sort_values(ascending = False).index ).head(20).plot.bar()
plt.ylabel('Change %')

plt.figure()
data['Change %'].reindex( data['Change %'].abs().sort_values(ascending = False).index ).head(20).sort_index().plot.bar()



#%% months with highest change (+ or -), using abs

# add new columns used to show better figure, 

# but time not evenly spaced not yet solved
data['Label'] = data.index.strftime('%Y %b')
plt.figure()
data.reindex( data['Change %'].abs().sort_values(ascending = False).index ).head(20).plot.bar(x = 'Label', y = 'Change %')

# need to sort_index
plt.figure()
data.reindex( data['Change %'].abs().sort_values(ascending = False).index ).head(20).sort_index().plot.bar(x = 'Label', y = 'Change %')
plt.ylabel('Change %')
plt.xlabel('')

plt.figure()
data.reindex( data['Change %'].abs().sort_values(ascending = False).index ).head(20).sort_index().plot(x = 'Label', y = 'Change %', linestyle = '', marker = 'o')

#%% make new series, then add NaN to missing date
df = data['Change %'].reindex( data['Change %'].abs().sort_values(ascending = False).index ).head(20).sort_index()

# resample method not working here (both year and month are missing)

"""
RESAMPLE

BMS: business month start, which might be different from the 1st

MS: month start
"""
plt.figure()
df.resample('BMS').asfreq().plot.bar()

plt.figure()
df.resample('MS').asfreq().plot(linestyle = '', marker = 'o', markersize = 5.)
plt.ylabel('Change %')
plt.ylim([-25.0,25.0])

#quarter start frequency
plt.figure()
df.resample('QS').asfreq().plot(linestyle = '', marker = 'o', markersize = 5.)
plt.ylabel('Change %')
plt.ylim([-25.0,25.0])

#quarter end frequency
plt.figure()
df.resample('AS').asfreq().plot(linestyle = '', marker = 'o', markersize = 5.)
plt.ylabel('Change %')
plt.ylim([-25.0,25.0])

#%% show average over year, or month

data.groupby(by = data.index.year).mean()

data.groupby(by = data.index.month).mean()

plt.figure()
data['Change %'].groupby(by = data.index.year).mean().plot.bar()

plt.figure()
data['Change %'].groupby(by = data.index.year).mean().loc[range(2005,2019)].plot.bar()
plt.ylabel('Change %')
plt.ylim([-4.0,4.0])


plt.figure()
data['Change %'].groupby(by = data.index.year).mean().plot(linestyle = '-')
plt.ylim([-4.0,4.0])

# plot describe() of change over time
plt.figure()
data['Change %'].groupby(by = data.index.year).describe()[ [u'mean', u'min', u'25%', u'50%', u'75%', u'max']] .plot(linestyle = '-', linewidth = 2.0)
plt.legend(loc = 'lower right')
plt.ylabel('Change %')
plt.ylim([-25.0,25.0])
# when copied in jupyternotebook, error thrown out.
# need to unstack the dataframe
plt.figure()
data['Change %'].groupby(by = data.index.year).describe().unstack()[ [u'mean', u'min', u'25%', u'50%', u'75%', u'max']] .plot(linestyle = '-', linewidth = 2.0)
plt.legend(loc = 'lower right')
plt.ylabel('Change %')
plt.ylim([-25.0,25.0])

plt.figure()
data['Change %'].groupby(by = data.index.year).describe().plot(linestyle = '-', marker = 'o', markersize = 5.)


plt.figure()
data['Price'].groupby(by = data.index.year).mean().plot.bar()


plt.figure()
data['Change %'].groupby(by = data.index.month).mean().plot.bar()
plt.xlabel('Month')
plt.ylim([-2.0,2.0])
#%%
plt.figure()
data.plot()

#%%
plt.figure()
data['Change %'].plot()


#%%
start_date = '2006-02-01'
end_date = '2018-02-01'
#%%
plt.figure()
data.loc[(data.index >= start_date) & (data.index <= end_date),'Change %'].plot()
#%%
plt.figure()
data['Change %'].plot.hist(bins = 100)

#%%
plt.figure()
data['Change %'].plot(kind = 'kde')
#%% 
plt.figure()
data.loc[(data.index >= start_date) & (data.index <= end_date),'Change %'].plot.hist(bins = 100)
#%% 
stats.shapiro(data.loc[:,'Change %'])

#%%
stats.kstest(data.loc[:,'Change %'],'norm')
#%%
stats.normaltest(data.loc[:,'Change %'])

#%%

stats.ttest_1samp(data.loc[:,'Change %'], data.loc[:,'Change %'].mean()  )
#%%
(test_statistic,p_value) = stats.shapiro(data.loc[(data.index >= start_date) & (data.index <= end_date),'Change %'])

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
data.loc[data.index.isin(pd.date_range('2005-02-01',end_date)), [u'Price', u'Open', u'High', u'Low']].plot()
plt.xlabel('')

#%%
plt.figure()
data.loc[(data.index >= start_date) & (data.index <= end_date), [u'Price', u'Open', u'High', u'Low']].plot()
#%%
plt.figure()
data.loc[data.index.isin(pd.date_range('2005-02-01',end_date)), [u'Price']].plot()
plt.xlabel('')

#%%
#%%
plt.figure()
data[ [u'Price', u'Change %']].plot()
#%%
plt.figure()
data.loc[ data.index.isin(pd.date_range('2010-02-01',end_date)), [u'Price', u'Change %']].plot()