#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:18:18 2018

@author: Chu Mai

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.datasets import load_boston

#%%
# Load and return the boston house-prices dataset 
boston = load_boston()

df = pd.DataFrame(data = boston.data, columns = boston.feature_names)

#%%
plt.figure(figsize = (16.0,12.0))
sns.heatmap(df.corr(),xticklabels= df.corr().columns.values,yticklabels=df.corr().columns.values, cmap = cm.PiYG, center = 0., annot = True, fmt = '.2f' )

#%%
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff


def plot_correlation_matrix(corr, xcols = None, ycols = None, filename = None, title = None):
    # corr is the correlation matrix obtained from a dataframe using pandas
    
    if xcols == None:
        xcols = corr.columns.tolist()
    if ycols == None:
        ycols = corr.columns.tolist()
    
    layout = dict(
        title = title,
        width = 800,
        height = 800,
#        margin=go.Margin(l=100, r=10, b=50, t=50, pad=5),
        margin=go.Margin(l=250, r=50, b=50, t=250, pad=4),
        yaxis= dict(tickangle=-30,
                    side = 'left',
                    ),
        xaxis= dict(tickangle=-30,
                    side = 'top',
                    ),
    )
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x= xcols,
        y= ycols,
        colorscale='Portland',
        reversescale=True,
        showscale=True,
        font_colors = ['#efecee', '#3c3636'])
    fig['layout'].update(layout)
    
    if filename == None:
        filename = 'correlation matrix'
    return py.plot(fig, filename= filename)

#%% 

plot_correlation_matrix( df.corr().round(2), filename = 'correlation matrix' )