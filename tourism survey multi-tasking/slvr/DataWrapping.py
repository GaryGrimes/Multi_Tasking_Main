# -*- coding: utf-8 -*-
"""
This script is to generate agent and network data for optimal tour (TTDP) solver.
@author: Kai Shen
Last modified on Nov. 15
"""
import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import progressbar as pb
from wordcloud import WordCloud
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic',
                               'Noto Sans CJK JP']


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
        Trips.trip_count += 1


def get_trip_chain(index):  # get trip info. for a specific tourist
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


def comp_extractor(_arr):
    for x in _arr:
        _row, _col = x // 10 - 1, x % 10 - 1
        yield comp_options[_row][_col]


# %% DATA PREPARATION
# read OD data
print('Reading OD data...\n')
OD_data = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', '観光客の動向調査.xlsx'), sheet_name='OD')
OD_data[['飲食費', '土産代']] = OD_data[['飲食費', '土産代']].replace(np.nan, 0).astype(int)

# %% socio-demographics
print('Reading personal info and socio-demographics data...')
ENQ2006 = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', '観光客の動向調査.xlsx'), sheet_name='ENQ2006')

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

# %% complaints and dissatisfaction
comp_freq_table = OD_data.loc[:, '不満点１':'不満点６'].values
temp = comp_freq_table.flatten()
comp_fre_array = temp[temp > 0]

y = np.bincount(comp_fre_array)
ii = np.nonzero(y)[0]

com_fre = list(zip(ii, y[ii]))

comp_options = [
    ['道路が混雑', '経路がわかりにくい', '道路が細い', '駐車場', '駐車の待ち時間', '駐車代'],
    ['バスの本数', '運賃', '道路が混雑', '乗り換え', 'バスの選択', 'バス停わかりにくい', 'バス運転', '車内で混雑'],
    ['電車の本数', '鉄道運賃', '乗り換え', '駅のバリアフリー', '駅がわからない', '電車内で混雑', '車内で混雑', '乗り換えが面倒']
]

g = comp_extractor(comp_fre_array)


with open(os.path.join(os.path.dirname(__file__), 'Database', 'comp_output.txt'), 'w') as f:
    for x in g:
        f.write(str(x) + ' ')

with open(os.path.join(os.path.dirname(__file__), 'Database', 'comp_output.txt'), 'r') as f:
    word_text = f.read()

# create word-cloud
font_path = "/System/Library/fonts/NotoSansCJKjp-Regular.otf"
wordcloud = WordCloud(font_path=font_path, regexp="[\w']+", background_color=None, mode='RGBA', scale=2,
                      colormap='magma')
wordcloud.generate(word_text)

# plot
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# similarly, bus one-day bus 不使用的理由之类的, 也可以同样做一个wordcloud.

# %% for each tourist (agent), generate an instance of the class
toursit_index = ENQ2006['ナンバリング'].values

flag = input('Generate and dump agent database? Press anything to continue, [enter] to skip.')
if flag:
    # configure pb settings
    p = pb.ProgressBar(widgets=[
        ' [', pb.Timer(), '] ',
        pb.Percentage(),
        ' (', pb.ETA(), ') ',
    ])

    p.start()
    total = ENQ2006.shape[0]

    agent_database = []
    transit_uid = []
    error_uid = []
    for count in range(ENQ2006.shape[0]):
        p.update(int((count / (total - 1)) * 100))

        if count % 100 == 0 and count:  # avoid print at 0
            print('\n----------  Parsing tourist: {} / {} ----------'.format(count, ENQ2006.shape[0]))
        x = Tourist(ENQ2006.loc[count, '整理番号'], ENQ2006.loc[count, 'ナンバリング'])
        trips_df = get_trip_chain(ENQ2006.loc[count, 'ナンバリング'])  # 这个方法好啊
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

        ''' skipping non-transit users here'''
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

    p.finish()
    file = open(os.path.join(os.path.dirname(__file__), 'Database', 'transit_user_database.pickle'), 'wb')
    # last modifed on Dec. 18. Duplicate users indentified and distinguished by indentical id
    pickle.dump(agent_database, file)

""" ok, DATA PREPARATION complete!
# read agent database
# with open(os.path.join(os.path.dirname(__file__), 'Database', 'transit_user_database.pickle'), 'rb') as file:
#     agent_database = pickle.load(file)

"""
# %% create Trip instances
flag = input('Generate trip database? Press anything to continue, [enter] to skip.')

if flag:
    # configure pb settings
    p = pb.ProgressBar(widgets=[
        ' [', pb.Timer(), '] ',
        pb.Percentage(),
        ' (', pb.ETA(), ') ',
    ])

    p.start()
    total = len(toursit_index)

    trip_database = []
    success_uid = []
    fail_uid = []
    for _idx, Uid in enumerate(toursit_index):
        p.update(int((_idx / (total - 1)) * 100))

        temp_df = get_trip_chain(Uid)
        try:
            # enumerate all trips
            origins, dests = temp_df['出発地'].values, temp_df['到着地'].values
            dep_hh, dep_mm = temp_df['出発時'].values, temp_df['出発分'].values
            arr_hh, arr_mm = temp_df['到着時'].values, temp_df['到着分'].values
            modes = temp_df.loc[:, '交通手段１':'交通手段６'].values
            Costs_food, Costs_sovn = temp_df['飲食費'].values, temp_df['土産代'].values
            for i in range(len(temp_df)):
                x = Trips(Uid, origins[i], dests[i])
                x.trip_index = i
                x.dep_time, x.arr_time = (dep_hh[i], dep_mm[i]), (arr_hh[i], arr_mm[i])
                x.mode = tuple(modes[i])
                x.food_cost, x.sovn_cost = Costs_food[i], Costs_sovn[i]
                trip_database.append(x)
            success_uid.append(Uid)
        except:
            fail_uid.append(Uid)
            pass

    p.finish()

    # test
    exist_usr = set(OD_data[OD_data.columns[2]].values)
    duplicate_usr = set([x for x in success_uid if success_uid.count(x) > 1])
    '''原始表的ENQ2006, OD sheet里有duplicate user的存在。修改了源表。
    将duplicate的user末尾加了两位以区别，treated as different user。'''

    # if to write into pickle file
    print('\n')
    flag_write = input('Dump trip database? Press anything to continue, [enter] to skip.')
    if flag_write:
        file = open(os.path.join(os.path.dirname(__file__), 'Database', 'trip_database.pickle'), 'wb')
        pickle.dump(trip_database, file)
