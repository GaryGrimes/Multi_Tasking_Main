# -*- coding: utf-8 -*-
"""
@author: Kai Shen
"""
import pandas as pd
import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from PTSurvey import PTSurvey as Pt
import shutil

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def assign_to_new_purpose(x):
    if x in [5, 6, 7, 16, 10, 12, 14, 15, 17]:
        return 5
    elif x == 8:
        return 6
    elif x == 9:
        return 7
    elif x == 11:
        return 8
    elif x == 13:
        return 9
    else:
        return x


# %% read OD data
print('Reading OD data...\n')
OD_data = pd.read_excel(os.path.join(os.path.dirname(__file__), '観光客の動向調査.xlsx'), sheet_name='OD')
OD_data[['飲食費', '土産代']] = OD_data[['飲食費', '土産代']].replace(np.nan, 0).astype(int)
# %% find complained routes frequency
Destination_names = list(range(1, 59)) + [99]
# create OD matrix
complained_OD_matrix = pd.DataFrame(np.zeros((59, 59)), index=Destination_names, columns=Destination_names, dtype=int)
frequency_OD_matrix = complained_OD_matrix.copy()

# trip frequency and 不满点 OD statistics
for i in range(len(OD_data)):
    frequency_OD_matrix.loc[OD_data.loc[i, '出発地'], OD_data.loc[i, '到着地']] += 1
    if OD_data.loc[i, '不満点１']:
        complained_OD_matrix.loc[OD_data.loc[i, '出発地'], OD_data.loc[i, '到着地']] += 1

# output if not in currrent directory
# if not os.path.exists(os.path.join(os.path.dirname(__file__), 'Database', 'frequency_OD_matrix.csv')):
#     frequency_OD_matrix.to_csv(os.path.join(os.path.dirname(__file__), 'Database', 'frequency_OD_matrix.csv'))
#     print('Saved frequency_OD_matrix to Database.')
# if not os.path.exists(os.path.join(os.path.dirname(__file__), 'Database', 'complained_OD_matrix.csv')):
#     complained_OD_matrix.to_csv(os.path.join(os.path.dirname(__file__), 'Database', 'complained_OD_matrix.csv'))
#     print('Saved complained_OD_matrix to Database.')
# %% Purpose statistics and correlation
print('Reading personal info and socio-demographics data...')
ENQ2006 = pd.read_excel(os.path.join(os.path.dirname(__file__), '観光客の動向調査.xlsx'), sheet_name='ENQ2006')
purpose_raw = ENQ2006.loc[:, ['整理番号', 'ナンバリング', '観光目的①', '観光目的②',
                              '観光目的③']]
# %% add more features
print('Calculating car trip dummy...')
col_name = ['交通手段１', '交通手段２', '交通手段３',
            '交通手段４', '交通手段５', '交通手段６', '交通手段７',
            '交通手段８', '交通手段９', '交通手段１０', '交通手段１１']
car_use = pd.Series(index=range(len(ENQ2006)))  # 是否小汽车或rental car
for i in range(len(ENQ2006)):
    car_use[i] = 1 if 1 in \
                      ENQ2006.loc[i, col_name].values or 2 in ENQ2006.loc[i, col_name].values else 0
# %% get person: purposes data
col_name = ['観光目的①', '観光目的②', '観光目的③']
# frequency(aggregate and disaggregate)
purpose_raw.loc[:, col_name] = purpose_raw.loc[:, col_name].replace(np.nan, 0).astype(int)
purpose = purpose_raw.loc[:, col_name]
purpose[col_name] = purpose.replace(np.nan, 0)[col_name].astype(int)
# %% Calculate purpose frequency

# 计算Purpose1&Purpose2别的频率
# size跟count的区别： size计数时包含NaN值，而count不包含NaN值
purpose_pattern_fre_by12 = purpose.groupby(by=col_name[0:2], as_index=False).size().reset_index(
    name='frequency')
# 计算Purpose1 & Purpose2 & Purpose3别的频率，不包括nan
purpose_pattern_fre_by123 = purpose.groupby(by=col_name, as_index=False).size().reset_index(name='frequency')
# %% Merged purposes (most significant for ANOVA after clustering):
# Cluster centroids: 1-6-8 ; 1-6 ; 5
print('Creating merged purposes..')
purpose_raw_merged = purpose_raw.copy()
purpose_raw_merged[col_name] = purpose

for i in col_name:
    purpose_raw_merged[i] = purpose_raw_merged[i].apply(lambda x: assign_to_new_purpose(x))

merged_purpose_pattern_fre_by123 = purpose_raw_merged[col_name].groupby(by=col_name, as_index=False).size().reset_index(
    name='frequency')

purpose_merged = purpose_raw_merged[col_name].apply(pd.value_counts).replace(np.nan, 0).astype(int)
purpose_merged.rename(index={0: 'None'}, inplace=True)
# %% person-destination set
# 4-25
print('Creating person: destination data...')
Cluster_cor = pd.read_excel(os.path.join(os.path.dirname(__file__), '観光客の動向調査.xlsx'), sheet_name='Correlation')
Cluster_cor.set_index(["ナンバリング"], inplace=True)

# Series既有数组的性质又有字典的性质

id_clst = Cluster_cor['Cluster']  # id和cluster对应的series

# cluster分别为：1,6,8 ; 1,6 ; 5

Dst_raw = OD_data.loc[:, ['ナンバリング', '出発地', '到着地']]
dst_res = Pt.create_dst_res(Dst_raw)
# %%  person - visited attraction areas
PT_list = Pt.write_PT_num(purpose_raw, dst_res)
# names of visited places
PT_list_semantic = Pt.write_PT_name(purpose_raw, dst_res)
# %%  统计car travel dummy, transit dummy 和 Kyoto stay dummy
print('Main person-destination database is "PT dummy" and "PT".')
PT_dummy = Pt.write_dummy(purpose_raw, dst_res)
PT_dummy['Clusters'] = np.array(id_clst)
#  更改column顺序
cols = list(PT_dummy)
cols.insert(1, cols.pop(cols.index('Clusters')))
PT_dummy = PT_dummy.loc[:, cols]

#  写destination dummy
# destination frequency统计量
PT_dummy = Pt.write_dest_dummy(PT_dummy)

means = PT_dummy.loc[:, '1':'59'].groupby([PT_dummy['Clusters'], PT_dummy['origin']]).mean()
counts = PT_dummy.loc[:, '1':'59'].groupby([PT_dummy['Clusters'], PT_dummy['origin']]).count()

means_origin = PT_dummy.loc[:, '1':'59'].groupby(PT_dummy['origin']).mean()
#  读取根据destination choice的clustering results

Dest_Clustering = scio.loadmat('slvr/Database/Dest_Clustering.mat')['Index']
D4 = Dest_Clustering[:, 3]
D6 = Dest_Clustering[:, 5]

PT_dummy['D4'] = D4
PT_dummy['D6'] = D6
#  更改columns顺序
cols = list(PT_dummy)
cols.insert(2, cols.pop(cols.index('D4')))
cols.insert(3, cols.pop(cols.index('D6')))
PT_dummy = PT_dummy.loc[:, cols]
#  写入excel。其中D4 D6的clustering是不可重现的，因为每次matlab的kmeans结果都不同。
#  第一次导出后的clustering结果请在excel中查看

# PT_dummy.to_excel('./Destination dummy.xlsx',sheet_name='Destinations')

# D4
D4_means = PT_dummy.loc[:, '1':'59'].groupby(PT_dummy['D4']).mean()
D4_counts = PT_dummy.loc[:, '1':'59'].groupby(PT_dummy['D4']).count()

# D6
D6_means = PT_dummy.loc[:, '1':'59'].groupby(PT_dummy['D6']).mean()
D6_counts = PT_dummy.loc[:, '1':'59'].groupby(PT_dummy['D6']).count()

# %%  Create Person-trip DB  (in PTSurvey Class)
# 建立一个新表没意义。用SQL一样的方式提取trips， modes和time吧PT_DB = pt.create_TP_DataBase(Data,PT_dummy,Dst_raw)

# Trip chains of every tourist

# places
Places = pd.read_excel(os.path.join(os.path.dirname(__file__), '観光客の動向調査.xlsx'), sheet_name='Place_code')
Place_names = dict(zip(Places.no, Places.name))  # places code 和 name的字典

# trips chains and travel times
Trips, Trip_len = Pt.create_trips(OD_data, PT_dummy)
Times, Travel_times = Pt.create_travel_time(OD_data, PT_dummy)

Modes = Pt.create_mode_split(OD_data, PT_dummy)
Costs = Pt.create_costs(OD_data, PT_dummy)

# 根据实际trip chain 添加car travel flag
car_flag = Pt.define_car_trip(PT_dummy, Modes, Trips)

PT = pd.concat([PT_dummy, car_flag['car_trip']], axis=1)
cols = list(PT)
cols.insert(6, cols.pop(cols.index('car_trip')))
PT = PT.loc[:, cols]

# # 根据实际trip chain 添加sightseeing bus travel flag
# sb_flag is modified to reflect both car trips (2) and sightseeing bus trips (1)
sb_flag = Pt.sightseeing_bus_trip(PT_dummy, Modes, Trips)

PT = pd.concat([PT, sb_flag['sb_trip']], axis=1)
cols = list(PT)
cols.insert(7, cols.pop(cols.index('sb_trip')))
PT = PT.loc[:, cols]

# dest. statistics in terms of car use
car_means = PT.loc[:, '1':'59'].groupby(PT['car_trip']).mean()
car_counts = PT.loc[:, '1':'59'].groupby(PT['car_trip']).count()

# dest. statistics in terms of sightseeing bus use
sb_means = PT.loc[:, '1':'59'].groupby(PT['sb_trip']).mean()
sb_counts = PT.loc[:, '1':'59'].groupby(PT['sb_trip']).count()

# %% 统计每个cluster的destination frequency
# 新建存放destination frequency的dataframe

print('Calculating destination frequency...')
des_fre = pd.DataFrame(index=range(1, 60), columns=[1, 2, 3])
des_fre.loc[:, [1, 2, 3]] = 0
# des_fre.columns.tolist() 获取列名

for i in dst_res.keys():
    visited = set(dst_res[i])  # 获取游客i的destination set
    cluster = id_clst[i]  # 获取游客i的cluster
    for k in visited:

        if k == 99: k = 59
        des_fre.loc[k, cluster] += 1

# dataframe转置
desfre_copy = des_fre.copy()  # 新建df存放百分比
for i in desfre_copy.columns:
    desfre_copy[i] = desfre_copy[i] / sum(id_clst == i)
# %% person_destination set DataFrame
pd_raw = pd.DataFrame(columns=['ナンバリング'] + list(range(1, 60)))
pd_raw['ナンバリング'] = purpose_raw['ナンバリング']
pdsum = 0


def create_pd(pd_raw, dst_res):  # construct person-destinations df
    for i in range(len(pd_raw)):
        try:
            res = [59 if x == 99 else x for x in list(dst_res[pd_raw['ナンバリング'][i]])]
            pd_raw.loc[i, res] = 1
        except:
            pass


create_pd(pd_raw, dst_res)
person_des = pd_raw.replace(np.nan, 0)

# %% PART 2
# print in the middle of screen
columns = shutil.get_terminal_size().columns
print('------------------------- PART 2 -------------------------'.center(columns))

# Calculate intrinsic utilities of attractions

# red leaves. Scores have been normalized by (I-Imin)/(Imax-Imin)
print('Calculating intrinsic utilities of attractions...')
red_leave_score = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Red leave.xlsx'), sheet_name='score')
red_leave_score.set_index(red_leave_score.Code, inplace=True)

shrine_temple_score = pd.DataFrame(index=red_leave_score.index,
                                   columns=['Code', 'name', 'density', 'popularity', 'average_score'])
shrine_temple_score[['Code', 'name']] = red_leave_score[['Code', 'name']]

# create temple and shrine dict
Temple_and_shrine = {}
for i in range(1, 38):
    file_name = str(i) + '.csv'
    temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Temple and shrine', file_name))
    if not temp.empty:
        Temple_and_shrine[i] = temp
    else:
        file_name = str(i) + '_wide.csv'
        temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Temple and shrine', file_name))
        Temple_and_shrine[i] = temp

# evaluate utility in temple and shrine dimension
for i in shrine_temple_score.index:
    temp = Temple_and_shrine[i]
    try:
        shrine_temple_score.loc[i, 'density'] = temp.shape[0]
        shrine_temple_score.loc[i, 'popularity'] = int(sum(temp['user_ratings_total']))
        shrine_temple_score.loc[i, 'average_score'] = np.dot(temp.rating, temp.user_ratings_total) / sum(
            temp.user_ratings_total)
        shrine_temple_score.loc[i, 'total_score'] = np.dot(temp.rating, temp.user_ratings_total)
    except Exception as e:
        print('Calculation error at attraction %d' % i)
        print('str(e):\t\t', str(e))

# shrine and temple log values
shrine_temple_score['log_score'] = np.log10(shrine_temple_score.total_score)
shrine_temple_score['log_score_normed'] = \
    (np.log10(shrine_temple_score.total_score) - min(np.log10(shrine_temple_score.total_score))) / \
    (max(np.log10(shrine_temple_score.total_score)) - min(np.log10(shrine_temple_score.total_score)))
# %% plot and show scores
print('Plotting temple and shrine popularity...')
to_plot = shrine_temple_score['total_score'].copy()
to_plot.sort_values(inplace=True, ascending=False)

fig = plt.figure(figsize=[5, 5], dpi=200)
ax0, ax1 = fig.add_subplot(211), fig.add_subplot(212)
ax0.barh(range(1, len(to_plot) + 1), to_plot, color='darkorange')

ax0.set_yticklabels([])
ax0.grid(linestyle=':', axis='x')
ax0.set(ylim=[0, 38])
ax0.invert_yaxis()  # labels read top-to-bottom
ax0.set_ylabel(ylabel='Areas', fontname='Times New Roman')
ax1.set_xlabel(xlabel='Popularity', fontname='Times New Roman')
ax0.set_title('Popularity in shrine and temples (ratings * # of reviews)', fontname='Times New Roman')

ax1.barh(range(1, len(to_plot) + 1), to_plot, log=True, color='darkorange')
ax1.set_yticklabels([])
ax1.set(ylim=[0, 38])
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_ylabel(ylabel='Areas', fontname='Times New Roman')
ax1.set_xlabel(xlabel='Popularity in log', fontname='Times New Roman')
ax1.grid(linestyle=':', axis='x')
# plt.savefig('Figure/Popularity in shrine and temples.png', bbox_inches='tight')
plt.show()

# %% Evaluate intrinsic utility in 'temple shrines and red leaves'
print('Evaluating intrinsic utility in "temple shrines and red leaves"')
ratio = sum(purpose_merged.loc[1, :]) / sum(purpose_merged.loc[6, :])  # 1:temple shrines 6:red leaves
Utility16 = {}
for i in range(1, 38):
    Utility16[i] = (ratio * shrine_temple_score.loc[i, 'log_score_normed'] + red_leave_score.loc[i, 'Final Score']) / (
            ratio + 1)

# plot final utility in 1, 6
temp = pd.Series(Utility16)
temp.sort_values(inplace=True, ascending=False)
print('Plotting utility score for temple-shrine and red leaves popularity...')
# plot scores
fig2 = plt.figure(figsize=(9, 5), dpi=200)

ax = fig2.add_subplot(111)
ax.barh(range(1, 38), temp.values, color='darkorange')

ax.set(ylim=[0, 38], xlim=[0, 0.9])
ax.set_yticks(list(range(1, 38, 1)))
ax.invert_yaxis()  # labels read top-to-bottom
ylabels = [Place_names[x] for x in temp.index]
ax.set_yticklabels(labels=ylabels, fontsize='x-small')
#
ax.grid(linestyle=':', axis='x')

ax.set_ylabel(ylabel='Area code', fontname='Times New Roman')
ax.set_title('Intrinsic utility in shrine-temples and red leaves', fontname='Times New Roman')
# plt.savefig('Figure/Attraction utility 16.png', bbox_inches='tight')
plt.show()

# %% グルメ　score
index, columns = range(1, 38), ['Code', 'name', 'count', 'food_count', 'average']
gourmet_score = pd.DataFrame(np.zeros([len(index), len(columns)]), index=index,
                             columns=columns, dtype=int)
gourmet_score[['Code', 'name']] = red_leave_score[['Code', 'name']]

for i in range(1, 38):
    gourmet_score.loc[i, 'count'] = sum(frequency_OD_matrix.loc[:, i])

for i in range(len(OD_data)):
    if OD_data.loc[i, '飲食費']:
        if OD_data.loc[i, '到着地'] in range(1, 38):
            gourmet_score.loc[OD_data.loc[i, '到着地'], 'food_count'] += 1
            gourmet_score.loc[OD_data.loc[i, '到着地'], 'average'] += OD_data.loc[i, '飲食費']

for i in range(1, 38):
    if gourmet_score.loc[i, 'food_count']:
        gourmet_score.loc[i, 'average'] = int(gourmet_score.loc[i, 'average'] / gourmet_score.loc[i, 'food_count'])

# TODO probability
gourmet_score['probability'] = gourmet_score['food_count'] / gourmet_score['count']

# TODO evaluate by # of restaurants in total and high-end restaurants
# %% グルメ　plot
prob = gourmet_score['probability'].copy()
prob.sort_values(inplace=True, ascending=False)
# plot scores
fig3 = plt.figure(dpi=200)

ax = fig3.add_subplot(111)
ax.barh(range(1, 38), prob.values)
ax.set_yticks(list(range(1, 38, 1)))
ylabels = [Place_names[x] for x in prob.index]
ax.set_yticklabels(labels=ylabels, fontsize='x-small')
ax.grid(linestyle=':', axis='x')
ax.set(ylim=[0, 38])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Probability of having food while visiting', fontname='Times New Roman')
# plt.savefig('Figure/Having food probability.png', bbox_inches='tight')
plt.show()

# %% utility in グルメ
ENQ2006.set_index('整理番号', inplace=True)
print('Evaluating utility in グルメ(gourmet)...')
# calculate total cost and average cost per person, if larger than 5,000
abs_count, avr_count = 0, 0
for i in range(len(OD_data)):
    if OD_data.loc[i, '飲食費']:
        if OD_data.loc[i, '到着地'] in range(1, 38):
            if OD_data.loc[i, '飲食費'] > 5000:
                abs_count += 1
            _ = OD_data.loc[i, '整理番号']
            try:
                GroupSize = ENQ2006.loc[_, '総人数']
            except KeyError:
                GroupSize = 1
            if GroupSize and ~np.isnan(GroupSize):
                if OD_data.loc[i, '飲食費'] / GroupSize > 5000:
                    avr_count += 1

food_count = sum(gourmet_score.food_count)

abs_ratio, avr_ratio = abs_count / food_count, avr_count / food_count

# %% read parsed # of (high-end) restaurants from OSM
Utility8 = pd.DataFrame(columns=['ordinary', 'high-end', 'utility'], index=range(1, 38)).astype(float)
osm_poi = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'Database', 'Gourmet and leisure',
                 'leisure and restaurants.csv')).set_index('index')

# parsing ordinary ln form ... add 1 to avoid division by zero
_ = np.log(osm_poi.restaurant + osm_poi.pub + osm_poi.bar + 1)
Utility8['ordinary'] = \
    (_ - min(_)) / (max(_) - min(_))

# high-end score
high_end_series = Utility8['high-end'].copy()
for i in range(1, 38):
    file_name = str(i) + '_wide'
    table = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'Database', 'Gourmet and leisure', 'high_end_restaurants',
                     '{}.csv'.format(file_name)))
    high_end_series[i] = len(table)

# ln form ... add 1 to avoid division by zero
_ = np.log(high_end_series + 1)
Utility8['high-end'] = \
    (_ - min(_)) / (max(_) - min(_))

# calculate utility
Utility8['utility'] = (Utility8['ordinary'] + abs_ratio * Utility8['high-end']) / (1 + abs_ratio)

# %% evaluate attraction utilities in leisure dimension
Utility5 = pd.DataFrame(columns=['facility', 'shop', 'utility'], index=range(1, 38)).astype(float)

# high-end score
_ = osm_poi.loc[:, 'museum':'art_center'].sum(axis=1)
Utility5['facility'] = (_ - min(_)) / (max(_) - min(_))

# parsing shops ln form ... add 1 to avoid division by zero
_ = np.log(osm_poi.shop + 1)
Utility5['shop'] = \
    (_ - min(_)) / (max(_) - min(_))

# balance scores in facility and shop dimensions, using ratio <- 0.5
ratio = 1
Utility5['utility'] = (Utility5['facility'] + ratio * Utility5['shop']) / (ratio + 1)
# %% Intrinsic utility matrix
IntrinsicUtility = pd.DataFrame(columns=['1_6', '8', '5'], index=range(1, 38))
IntrinsicUtility['1_6'] = Utility16.values()
IntrinsicUtility['8'] = Utility8['utility']
IntrinsicUtility['5'] = Utility5['utility']

# IntrinsicUtility.to_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Intrinsic Utility.csv'))

# TODO utility plot

print('Code complete.')

# %% extract trip chain and day of travel

PT_day = PT.loc[:, 'ナンバリング':'Kyoto stay dummy']
day_of_travel = ENQ2006.loc[:, ['ナンバリング', '旅行日程']].reset_index()
PT_day['day of travel'] = day_of_travel['旅行日程']
