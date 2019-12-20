# -*- coding: utf-8 -*-
"""
@author: Kai Shen
"""
import pandas as pd
import os
import numpy as np
import pickle
from scipy import stats
import seaborn as sns
import scipy.io as scio
import matplotlib.pyplot as plt
from PTSurvey import PTSurvey as Pt
import shutil

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_trip(index):
    try:
        # raw data returned
        segment = OD_data[OD_data['ナンバリング'] == index].reset_index(drop=True)  # 去掉Index索引哦
        if not segment.empty:
            # return cleaned data (merge records with identical origin and destination)
            journey = pd.DataFrame(columns=OD_data.columns[1:-3])

            # create a np array to store into the DataFrame named 'journey'
            i, j = 0, 0  # j is index for new DF 'journey
            while i < len(segment):
                start = int(i)
                # loc是通过标签索引，iloc是通过偏移量（位置）索引，所以不能用str对列索引
                while segment.loc[i, '出発地'] == segment.loc[i, '到着地'] and i + 1 < len(segment):
                    i += 1
                end = int(i)
                # origin and destination
                journey.loc[j, '出発地':'到着地'] = [segment.loc[start, '出発地'], segment.loc[end, '到着地']]
                # departure and arrival time
                journey.loc[j, '出発時':'到着分'] = segment.loc[end, '出発時':'到着分']
                # read travel means
                means = []
                for k in range(start, end + 1):
                    means.extend([foo for foo in segment.loc[k, '交通手段１':'交通手段６'] if foo > 0])
                # means 补全
                if len(means) > 6:
                    means = means[0:6]
                means.extend([0 for _ in range(6 - len(means))])
                journey.loc[j, '交通手段１':'交通手段６'] = means

                # read 不満点
                dissatis = []
                for k in range(start, end + 1):
                    dissatis.extend([foo for foo in segment.loc[k, '不満点１':'不満点６'] if foo > 0])
                # dissatis 补全
                if len(dissatis) > 6:
                    dissatis = dissatis[0:6]
                dissatis.extend([0 for _ in range(6 - len(dissatis))])
                journey.loc[j, '不満点１':'不満点６'] = dissatis
                # costs
                journey.loc[j, '飲食費'] = sum(segment.loc[start:end, '飲食費'])
                journey.loc[j, '土産代'] = sum(segment.loc[start:end, '土産代'])
                i += 1
                j += 1
                pass
            journey['整理番号'], journey['ナンバリング'] = segment['整理番号'][0], segment['ナンバリング'][0]
            journey['トリップ番号'] = list(range(1, j + 1))
            # replace nan and convert to int
            journey.replace(np.nan, 0, inplace=True)
            journey = journey.astype(int)
            return journey
        else:
            # print('Found no results for current id: {}.'.format(index))
            return None
    except NameError:
        print('Please make sure OD_data exists in the variable list')
        return None


class Agent(object):
    agent_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Agent.agent_count += 1


# %% DATA PREPARATION
# read OD data
print('Reading OD data...\n')
OD_data = pd.read_excel(os.path.join(os.path.dirname(__file__), '観光客の動向調査.xlsx'), sheet_name='OD')
OD_data[['飲食費', '土産代']] = OD_data[['飲食費', '土産代']].replace(np.nan, 0).astype(int)

# %% socio-demographics
print('Reading personal info and socio-demographics data...')
ENQ2006 = pd.read_excel(os.path.join(os.path.dirname(__file__), '観光客の動向調査.xlsx'), sheet_name='ENQ2006')

# %% dwell time data
Dwell_time = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Dwell time array.xlsx'))

# check and clean average dwell time
Dwell_time.loc[35, 'mean'] = Dwell_time['mean'][Dwell_time['mean'] != 5].mean()

# %% tourist preference duplicate ナンバリング.
Preference = {}
Cluster_probability = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Cluster probability.xlsx'))
for _idx in range(Cluster_probability.shape[0]):
    index, u_prob = Cluster_probability.loc[_idx, '整理番号'], np.array(Cluster_probability.loc[_idx, 'EST1':'EST3'])
    if np.isnan(u_prob[0]):
        u_pre = None
    else:
        u_pre = np.array([u_prob[0] + u_prob[1], u_prob[1], u_prob[2]]) / (u_prob[0] + 2 * u_prob[1] + u_prob[2])
    Preference[index] = u_pre
# %%  attraction area utilities
Intrinsic_utilities = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Intrinsic Utility.xlsx'),
                                    sheet_name='data')
UtilMatrix = []
for _idx in range(Intrinsic_utilities.shape[0]):
    temp = np.around(list(Intrinsic_utilities.iloc[_idx, 1:4]), decimals=3)
    UtilMatrix.append(temp)
# %% edge utility data (time and cost)
# Edge_time_matrix = pd.read_csv(
#    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'wide_transit_time_matrix.csv'), index_col=0)
Edge_time_matrix = pd.read_excel(
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'transit_time_update2.xlsx'), index_col=0)
# rename columns from str to int
Edge_time_matrix.columns = range(Edge_time_matrix.shape[0])
#  check principle of optimality of travel time

# %% need several iterations to make sure direct travel is shorter than any detour
NoUpdate, itr = 0, 0
while not NoUpdate:
    print('Current iteration: {}'.format(itr + 1))
    NoUpdate = 1
    for i in range(Edge_time_matrix.shape[0] - 1):
        for j in range(i + 1, Edge_time_matrix.shape[0]):
            time = Edge_time_matrix.loc[i, j]
            shortest_node, shortest_time = 0, time
            for k in range(Edge_time_matrix.shape[0]):
                if Edge_time_matrix.loc[i, k] + Edge_time_matrix.loc[k, j] < shortest_time:
                    shortest_node, shortest_time = k, Edge_time_matrix.loc[i, k] + Edge_time_matrix.loc[k, j]
            if shortest_time < time:
                NoUpdate = 0
                print('travel time error between {0} and {1}, shortest path is {0}-{2}-{1}'.format(i, j, shortest_node))
                Edge_time_matrix.loc[j, i] = Edge_time_matrix.loc[i, j] = shortest_time
    itr += 1
    if NoUpdate:
        print('Travel time update complete.')
# %%
# Edge_time_matrix = Edge_time_matrix / 60
# output to database (Final)
# Edge_time_matrix.to_excel(
#     os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'transit_time_wide_update.xlsx'))
Edge_cost_matrix = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'wide_transit_fare_matrix.csv'), index_col=0)
Edge_cost_matrix.columns = range(Edge_cost_matrix.shape[0])

# %% wrapping distance data for path penalty function
Edge_dist_matrix = pd.read_excel(
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'driving_wide_distance_matrix.xlsx'),
    index_col=0)
# rename columns from str to int
Edge_dist_matrix.columns = range(Edge_dist_matrix.shape[0])
#  check principle of optimality of travel distance
# need several iterations to make sure direct travel is shorter than any detour
NoUpdate, itr = 0, 0
while not NoUpdate:
    print('Current iteration: {}'.format(itr + 1))
    NoUpdate = 1
    for i in range(Edge_dist_matrix.shape[0] - 1):
        for j in range(i + 1, Edge_dist_matrix.shape[0]):
            dist = Edge_dist_matrix.loc[i, j]
            shortest_node, shortest_dist = 0, dist
            for k in range(Edge_dist_matrix.shape[0]):
                if Edge_dist_matrix.loc[i, k] + Edge_dist_matrix.loc[k, j] < shortest_dist:
                    shortest_node, shortest_time = k, Edge_dist_matrix.loc[i, k] + Edge_dist_matrix.loc[k, j]
            if shortest_dist < dist:
                NoUpdate = 0
                print('travel time error between {0} and {1}, shortest path is {0}-{2}-{1}'.format(i, j, shortest_node))
                Edge_dist_matrix.loc[j, i] = Edge_dist_matrix.loc[i, j] = shortest_dist
    itr += 1
    if NoUpdate:
        print('Travel distance update complete.')

# %% for each tourist (agent), generate an entity of a class
agent_database = []
transit_uid = []
error_uid = []
for count in range(ENQ2006.shape[0]):
    if count % 100 == 0 and count:
        print('\n----------  Parsing tourist: {} / {} ----------'.format(count, ENQ2006.shape[0]))
    x = Agent(ENQ2006.loc[count, '整理番号'], ENQ2006.loc[count, 'ナンバリング'])
    trips_df = get_trip(ENQ2006.loc[count, 'ナンバリング'])
    # get preference
    x.preference = Preference[ENQ2006.loc[count, '整理番号']]
    # get o,d and observed path
    if trips_df is None:
        continue
    trips_df = pd.DataFrame(trips_df)  # define type
    path = [trips_df['出発地'][0]] + list(trips_df['到着地'])
    if len(path) < 3:
        print(ValueError('path length < 2, at {}'.format(ENQ2006.loc[count, 'ナンバリング'])))
        error_uid.append(ENQ2006.loc[count, 'ナンバリング'])
    mode_choices = np.array(trips_df.loc[:, '交通手段１':'交通手段５'])
    # skip non-transit users
    if path[0] not in range(1, 48) or path[-1] not in range(1, 48) or 6 in mode_choices or 7 in mode_choices:
        continue
    x.trip_df = trips_df
    x.path_obs = path
    # get time budget
    T_start_h, T_end_h = trips_df['出発時'].iloc[0], trips_df['到着時'].iloc[-1]
    T_start_m, T_end_m = trips_df['出発分'].iloc[0], trips_df['到着分'].iloc[-1]
    # autofill start time using travel time matrix
    if not T_start_h:
        # if exists arrival time at the first destination
        if trips_df.loc[0, '到着時']:
            try:
                _travel_time = Edge_time_matrix.loc[path[0] - 1, path[1] - 1]  # try get travel time
            except KeyError:
                _travel_time = Edge_time_matrix[path[0] - 1][
                    Edge_time_matrix[path[0] - 1] > 0].mean()  # skip same origin

            T_start = trips_df.loc[0, '到着時'] * 60 + trips_df.loc[0, '到着分'] - _travel_time
            if T_start < 0:
                raise ValueError(
                    'Travel time error, please check trip start time. id: {}'.format(ENQ2006.loc[count, 'ナンバリング']))
            else:
                T_start_h, T_start_m = T_start // 60, T_start % 60
        else:
            print('cannot find start time for person: {}'.format(ENQ2006.loc[count, 'ナンバリング']))
            continue

    # autofill end time using travel time matrix
    if not T_end_h:
        # if exists departure time from the second last destination
        if trips_df.loc[trips_df.shape[0] - 1, '出発時']:
            try:
                _travel_time = Edge_time_matrix.loc[path[-2] - 1, path[-1] - 1]
            except KeyError:
                _travel_time = Edge_time_matrix[path[-1] - 1][
                    Edge_time_matrix[path[-1] - 1] > 0].mean()  # skip same origin

            T_end = trips_df.loc[trips_df.shape[0] - 1, '出発時'] * 60 + trips_df.loc[
                trips_df.shape[0] - 1, '出発分'] + _travel_time
            if T_end < 0:
                raise ValueError(
                    'Travel time error, please check trip end time. id: {}'.format(ENQ2006.loc[count, 'ナンバリング']))
            else:
                T_end_h, T_end_m = T_end // 60, T_end % 60
        else:
            print('cannot find start time for person: {}'.format(ENQ2006.loc[count, 'ナンバリング']))
            continue

    T_start, T_end = T_start_h * 60 + T_start_m, T_end_h * 60 + T_end_m
    if T_end < T_start:
        raise ValueError('Time cost error, at {}'.format(ENQ2006.loc[count, 'ナンバリング']))
    else:
        x.time_budget = T_end - T_start
    agent_database.append(x)
    # record transit user id
    transit_uid.append(ENQ2006.loc[count, 'ナンバリング'])

file = open(os.path.join(os.path.dirname(__file__), 'Database', 'agent_database.pickle'), 'wb')
pickle.dump(agent_database, file)

""" ok, DATA PREPARATION complete!
# read agent database
# with open(os.path.join(os.path.dirname(__file__), 'Database', 'agent_database.pickle'), 'rb') as file:
#     agent_database = pickle.load(file)

"""
