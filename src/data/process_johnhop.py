#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:17:34 2020

@author: prapul
"""

import pandas as pd
import numpy as np

from datetime import datetime

def relational_John_hopkins():
    data_path='../../data/raw/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    full_data=pd.read_csv(data_path)
    
    pd_data_base=full_data.rename(columns={'Country/Region':'country',
                      'Province/State':'state'})
    pd_data_base['state']=pd_data_base['state'].fillna('no')  
    
    pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)
    
    pd_relational=pd_data_base.set_index(['state','country']) \
                                .T                              \
                                .stack(level=[0,1])             \
                                .reset_index()                  \
                                .rename(columns={'level_0':'date',
                                                   0:'confirmed'},
                                                  )
    pd_relational['date']=pd_relational.date.astype('datetime64[ns]')
    
    pd_relational.to_csv('../../data/processed/COVID_relational_confirmed.csv',sep=';',index=False)
    print("Relational model prepared")
    
if __name__ == '__main__':

    relational_John_hopkins()
    