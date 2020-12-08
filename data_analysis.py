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
       'Year', 'Month', 'Day of Week', 'Day', 'Hour']

data_pd = data_load.groupby(col)['Volume'].sum().reset_index()

data_pd['Date'] = pd.to_datetime(data_pd[['Year', 'Month', 'Day']])
data_pd.index = data_pd['Datetime']

data_pd_0 = data_pd[data_pd['Direction'] == 0].sort_values(['Year', 'Month', 'Day'])

plt.figure(0)
data_pd_0[(data_pd_0['Date'] >= '2018-07-09') & (data_pd_0['Date'] <= '2018-08-10')]['Volume'].plot(label='Mois du 09/07/18 au 10/08/18')
data_pd_0[(data_pd_0['Date'] >= '2018-07-09') & (data_pd_0['Date'] <= '2018-07-15')]['Volume'].plot(label='Semaine 09/07 du 15/07')
data_pd_0.loc['2018-07-16', 'Volume'].plot(label='Journée du 16/07')
plt.ylabel("Volume")
plt.title("Du 09/07/18 au 10/08/18")
plt.legend()
plt.show()
