# -*- coding: utf-8 -*-
"""
This script is to generate statistics for observed visit and trip frequency.
Also used to evaluate the simualtion results .
Last modified on Dec. 18
"""
import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import progressbar as pb

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


class Trips(object):
    trip_count = 0

    def __init__(self, uid, origin, dest):
        self.uid = uid
        self.o, self.d = origin, dest
        # self.first_trip_dummy = 0  # whether is the first trip of current tourist
        self.trip_index = None
        self.dep_time, self.arr_time = (0, 0), (0, 0)  # time formatted as tuple
        self.mode = None  # if not empty, formmated into a tuple with at most 6 candidates
        self.food_cost, self.sovn_cost = None, None


if __name__ == '__main__':
    # read agent database
    with open(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'transit_user_database.pickle'), 'rb') as file:
        agent_database = pickle.load(file)  # note: agent = tourists here

    with open(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'trip_database.pickle'), 'rb') as file:
        trip_database = pickle.load(file)  # note: agent = tourists here

    # %% statistics of interest