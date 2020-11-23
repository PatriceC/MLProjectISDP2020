# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:54:58 2020

@author: Patrice CHANOL
"""

import numpy as np
import pandas as pd
import torch

# %% Load Data

data_load = pd.read_csv('./Radar_Traffic_Counts.csv')
data_load = data_load.drop(columns=['Time Bin','location_name'])
data_load['Direction'] = data_load['Direction'].astype('category').cat.codes


# %% Select set

def main(data=data_load, time_thin='Hour', series=6, pred=1, batch_size=128, train_per_length=0.8):

    if time_thin == 'Hour':
        col = ['Direction', 'location_latitude', 'location_longitude',
               'Year', 'Month', 'Day of Week', 'Day']
        data_pd = data.groupby(col + [time_thin])['Volume'].sum().reset_index()
        data_pd = data.pivot_table(index=col, columns=time_thin,
                                    values='Volume', fill_value=0).reset_index()
    elif time_thin == 'Day':
        col = ['Direction', 'location_latitude', 'location_longitude',
               'Year', 'Month', 'Day of Week']
        data_pd = data.groupby(col + [time_thin])['Volume'].sum().reset_index()
        data_pd = data.pivot_table(index=col, columns=time_thin,
                                    values='Volume', fill_value=0).reset_index()
    elif time_thin == 'Month':
        col = ['Direction', 'location_latitude', 'location_longitude',
               'Year']
        data_pd = data.groupby(col + [time_thin])['Volume'].sum().reset_index()
        data_pd = data.pivot_table(index=col, columns=time_thin,
                                    values='Volume', fill_value=0).reset_index()
    elif time_thin == 'Year':
        col = ['Direction', 'location_latitude', 'location_longitude']
        data_pd = data.groupby(col + [time_thin])['Volume'].sum().reset_index()
        data_pd = data.pivot_table(index=col, columns=time_thin,
                                    values='Volume', fill_value=0).reset_index()
