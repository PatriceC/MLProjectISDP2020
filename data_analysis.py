# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:19:45 2020

@author: Patrice CHANOL
"""

import pandas as pd
import matplotlib.pyplot as plt

# %% Load Data

data_load = pd.read_csv('./Radar_Traffic_Counts.csv')
data_load = data_load.drop(columns=['Time Bin', 'location_name'])
data_load['Direction'] = data_load['Direction'].astype('category').cat.codes


# %% Select set

col = ['Direction', 'location_latitude', 'location_longitude',
       'Year', 'Month', 'Day of Week', 'Day','Hour']

data_pd = data_load.groupby(col)['Volume'].sum().reset_index()

data_pd['Datetime'] = pd.to_datetime(data_pd[['Year', 'Month', 'Day','Hour']])
data_pd.index = data_pd['Datetime']

data_pd_0 = data_pd[data_pd['Direction'] == 0].sort_values(['Year', 'Month', 'Day'])

data_pd_0.loc['2018-07','Volume'].plot()
data_pd_0.loc['2018-07-19','Volume'].plot()
plt.show()
data_pd_0.loc['2018-07-19','Volume'].plot()
plt.show()
