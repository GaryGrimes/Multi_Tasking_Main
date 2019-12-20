# -*- coding: utf-8 -*-
"""
@author: Kai Shen
"""
import pandas as pd
import os
import numpy as np
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
            print('Found no results for current id: {}.'.format(index))
            return None
    except NameError:
        print('Please make sure OD_data exists in the variable list')
        return None



# %% read OD data

print('Reading OD data...\n')
OD_data = pd.read_excel(os.path.join(os.path.dirname(__file__), '観光客の動向調査.xlsx'), sheet_name='OD')
OD_data[['飲食費', '土産代']] = OD_data[['飲食費', '土産代']].replace(np.nan, 0).astype(int)

# %% Purpose statistics and correlation
print('Reading personal info and socio-demographics data...')
ENQ2006 = pd.read_excel(os.path.join(os.path.dirname(__file__), '観光客の動向調査.xlsx'), sheet_name='ENQ2006')
purpose_raw = ENQ2006.loc[:, ['整理番号', 'ナンバリング', '観光目的①', '観光目的②',
                              '観光目的③']]
# %% dwell time statistics
# TODO 建一个dict存放 Ti （每个attraction的dwell time）
# 设置正态分布参数，其中loc是期望值参数，scale是标准差参数
T = np.array([90, 56, 77, 85, 100, 79, 83])
mean, std = T.mean(), T.std()

X = stats.norm(loc=mean, scale=std)

# 计算随机变量的期望值和方差
print(X.stats())

# 对随机变量取10000个值
x = X.rvs(size=10000)
print(np.mean(x), np.var(x))

plt.hist(x, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("区间")
# 显示纵轴标签
plt.ylabel("频数/频率")
# 显示图标题
plt.title("频数/频率分布直方图")
plt.show()

# %% dwell time box plot
Dwell_time_arrays = {}

# each attraction as a list
for _idx, _numbering in enumerate(OD_data.ナンバリング.unique()):
    _size = len(OD_data.ナンバリング.unique())
    if _idx % 200 == 0:
        print('Parsing {}/{} of all tourists...'.format(_idx + 1, _size))
    df_temp = get_trip(_numbering)
    # go through all trips
    for _row in range(df_temp.shape[0]):
        if df_temp.loc[_row, '到着地'] in range(1, 38) and _row < df_temp.shape[0] - 1:
            _place = df_temp.loc[_row, '到着地']
            if df_temp.loc[_row + 1, '出発時'] and df_temp.loc[_row, '到着時']:  # avoid zeros
                _dwell_time = df_temp.loc[_row + 1, '出発時'] * 60 + df_temp.loc[_row + 1, '出発分'] - (
                        df_temp.loc[_row, '到着時'] * 60 + df_temp.loc[_row, '到着分'])
                if 0 < _dwell_time < 720:
                    if _place not in Dwell_time_arrays:
                        Dwell_time_arrays[_place] = []
                    Dwell_time_arrays[_place].append(_dwell_time)
                else:
                    continue

# %% make box plot
area, dwell_time = [], []
for key in Dwell_time_arrays:
    for item in Dwell_time_arrays[key]:
        area.append(key)
        dwell_time.append(item)

df_data = pd.DataFrame({'area': area, 'dwell_time': dwell_time})
# df_data.to_csv(os.path.join(os.path.dirname(__file__), 'Dwell time.csv'))

fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(1, 1, 1)

sns.set(style="whitegrid")
sns.boxplot(y='area', x='dwell_time', data=df_data, showfliers=False, ax=ax2, orient='h', palette="vlag")
plt.show()

# %% into arrays
Dwell_statistics = []
for area in range(1, 38):
    data = Dwell_time_arrays[area]
    # min, median, mean, max
    pnt = [np.min(data), np.median(data), np.mean(data), np.max(data)]
    Dwell_statistics.append(pnt)

Dwell = pd.DataFrame(Dwell_statistics, index=range(1, 38), columns=['min', 'median', 'mean', 'max'])

# # %% food and souvenir cost statistics
# cost_food, cost_sove = [], []
# for _numbering in ENQ2006['ナンバリング']:
#     df_temp = get_trip(_numbering)
#     if df_temp is not None:
#         _cost_food, _cost_souv = sum(df_temp['飲食費']), sum(df_temp['土産代'])
#         cost_food.append(_cost_food)
#         cost_sove.append(_cost_souv)
#
#     else:
#         cost_food.append(np.nan)
#         cost_sove.append(np.nan)
#
# ENQ2006['food_cost'] = cost_food
# ENQ2006['sove_cost'] = cost_sove
