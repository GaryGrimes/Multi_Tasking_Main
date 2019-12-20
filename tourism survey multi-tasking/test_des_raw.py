#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:12:00 2019

@author: gary
"""

#%% person_destination set DataFrame
pd_raw = pd.DataFrame(columns = ['ナンバリング']+list(range(1,60)))    
pd_raw['ナンバリング'] = Purpose_raw['ナンバリング']
pdsum = 0
def create_pd(pd_raw, dst_res):  # construct person-destinations df
    for i in range(len(pd_raw)):
        try:
            res = list(dst_res[pd_raw['ナンバリング'][i]])
            pd_raw.loc[i,res] = 1
        except:
            pass

create_pd(pd_raw, dst_res)