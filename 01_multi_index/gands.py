#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:40:47 2018

@author: chumai
"""

import pandas as pd
import copy as cp
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8.0, 6.0 # set figure size
plt.rcParams['lines.linewidth'] = 5.0 # set default linewidth
#%%
df = pd.read_excel('gands.xls', skiprows = range(5), skipfooter = 3, index_col = 0)

df.info()
df.head()
df.tail()
#%%
l1 = ['Balance', 'Exports', 'Imports']
l2 = ['Total', 'Goods', 'Services']
tuples = [ (i,j) for i in l1 for j in l2 ]
colnames = pd.MultiIndex.from_tuples(tuples)

df = pd.DataFrame( df.values[:,0:9],  index = df.index, columns = colnames, dtype = 'float' )

df.info()
#%%
for col in l1:
    plt.figure()
    df.loc[:,col].plot(linewidth = 5.0)
    plt.ylabel(col +' (Millions USD)')
    
    plt.savefig(col+'.png', bbox_inches = 'tight', dpi = 400)

#%%
for col in l1:
    plt.figure()
    df.loc[:,col].plot.bar()
    plt.ylabel(col)
    plt.show()
#%% ratio imports/exports
df2 = df.loc[:,'Imports']/df.loc[:,'Exports']
plt.figure()
df2.plot()
plt.ylabel('Imports / Exports')

plt.savefig('imports_vs_exports.png', bbox_inches = 'tight', dpi = 400)
#%%
plt.figure()
df2.plot.bar()
plt.ylabel('Imports / exports')

#%% balance not exactly well done
df.loc[:,'Balance'] / (df.loc[:,'Exports'] - df.loc[:,'Imports'])

#%%
data = pd.read_excel('US_population.xls',sheetname = 'Data', skiprows = 3)
df3 = data.loc[data['Country Code'] == 'USA', data.columns[5:] ]

df3 = df3.melt(var_name = 'Period', value_name = 'Population')

df3.set_index('Period', inplace = True)
# IMPORTANT: indices must be of same type as in df for later merging of two dataframes. Try otherwise, you will see what happens
df3.index = df3.index.astype('Int64')


df3.info()
#%%
plt.figure()
(1.0e-6*df3).plot()
plt.ylabel('Population (millions)')

plt.savefig('population.png', bbox_inches = 'tight', dpi = 400)

#%% merge two dataframes using indices
df5 = df.merge(df3, how = 'outer', left_index =True, right_index = True )
# assign multi-index
cols = pd.MultiIndex.from_tuples(df.columns.tolist() +  [ ('Census', 'Population')]) 
df5.columns = cols

df5.info()
df5.head()

df5.tail()
#%% the same thing done here

df5 = pd.concat([df,df3], axis = 1, join = 'outer')
df5.columns = cols

df5.info()
df5.head()
df5.tail()


#%% also

df6 = cp.deepcopy(df)
df6['Census','Population'] = df3['Population']

df6.info()
df6.head()
df6.tail()
#%% drop a column from multi-index columns

df6.drop('Population', axis = 1, level = 1, inplace = True)

df6.drop('Total', axis = 1, level = 1)

#%% divide each columns by the population
l1 = ['Balance', 'Exports', 'Imports']
for col in l1:
    df7 = df5[col].divide(df5['Census', 'Population'], axis = 'index') * 1.0e6 # convert into USD
    plt.figure()
    df7.plot()
    plt.ylabel(col + ' per capita (USD)')
    
    plt.savefig(col+'_per_capita.png', bbox_inches = 'tight', dpi = 400)




