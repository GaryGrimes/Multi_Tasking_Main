# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:20:14 2018

@author: Kai Shen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from PTSurvey import PTSurvey as pt

# %% read OD data
OD_data = pd.read_excel('観光客の動向調査.xlsx', sheetname='OD')

Dest_No = list(range(1, 59)) + [99]

OD_Table = pd.DataFrame(np.zeros((59, 59)), index=Dest_No, columns=Dest_No)

# Data.iloc[len(Data)-1, 1]  # 索引数据

Raw = OD_data.loc[:, ['出発地', '到着地', '不満点１', '不満点２',
                      '不満点３', '不満点４', '不満点５', '不満点６']]
Raw_cld = Raw[Raw['不満点１'] != 0]

# 不满点OD统计
for index in range(len(Raw_cld)):
    OD_Table.loc[Raw_cld.iloc[index, 0], Raw_cld.iloc[index, 1]] += 1
print('Dissatisfaction frequency(origin based)')
print(OD_Table.sum(axis=1))  # 按行统计

# OD_Table.to_excel('foo.xlsx', sheet_name='Sheet1')

# %% Purpose statistics and correlation
ENQ2006 = pd.read_excel('./観光客の動向調査.xlsx', sheet_name='ENQ2006')
Purpose_raw = ENQ2006.loc[:, ['整理番号', 'ナンバリング', '観光目的①', '観光目的②',
                              '観光目的③']]
# %% 数据处理 --> add more features
col_n = ['交通手段１', '交通手段２', '交通手段３',
         '交通手段４', '交通手段５', '交通手段６', '交通手段７',
         '交通手段８', '交通手段９', '交通手段１０', '交通手段１１']
car_use = pd.Series(index=range(len(ENQ2006)))  # 是否小汽车或rental car
for i in range(len(ENQ2006)):
    car_use[i] = 1 if 1 in \
                      ENQ2006.loc[i, col_n].values or 2 in ENQ2006.loc[i, col_n].values else 0

# %% 计算每个目的的频率
col_n = ['観光目的①', '観光目的②', '観光目的③']
# frequency(aggregate and disaggregate)
Purpose_raw.loc[:, col_n] = Purpose_raw.loc[:, col_n].replace(np.nan, 0).astype(int)
Freq_sel = Purpose_raw.loc[:, col_n]
# Freq_da = Purpose_raw.replace(np.nan,0).apply(pd.value_counts)
# Freq_a = Freq_da.replace(np.nan,0).sum(axis=1)
#
# Freq_sel['count'] = 1
# %% 计算每条记录的频率
# Freq_sel["Purpose_all"] = Freq_sel[col_n[0]].map(str) + Freq_sel[col_n[1]].map(str) + Freq_sel[col_n[2]].map(str)

# 计算Purpose1&Purpose2别的频率。  size跟count的区别： size计数时包含NaN值，而count不包含NaN值
Records_12 = Freq_sel.groupby(by=col_n[0:2], as_index=False).size().reset_index(
    name='Freq')  # Records_12.rename(columns ={'観光目的③':'freq'}, inplace = True)

# 校对1=1&2=17时候的频率。都为54，正确
Freq_sel.loc[(Freq_sel['観光目的②'] == 17) & (Freq_sel[
                                              '観光目的①'] == 1)].count()  # A common operation is the use of boolean vectors to filter the data. The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses.

# 计算Purpose1&Purpose2&Purpose3别的频率，不包括nan
Records_123 = Freq_sel.groupby(by=col_n, as_index=False).size().reset_index(name='Freq')

# %% 因为pandas在groupby时不包括nan，所以把nan替换为0
new_Records_12 = Freq_sel.replace(np.nan, 0).groupby(by=col_n[0:2], as_index=False).size().reset_index(name='Freq')
new_Records_123 = Freq_sel.replace(np.nan, 0).groupby(by=col_n, as_index=False).size().reset_index(name='Freq')


# %% 增加三列，替换原先的17类作为新的三类category
# 
def fun(x):
    if x == 0:
        return 0
    if x in [3, 8, 6, 9]:
        return 1
    elif x in [1, 4, 5]:
        return 2
    elif x == 17:
        return 4
    else:
        return 3


# 如何多列 apply？
Freq_sel[col_n] = Freq_sel.replace(np.nan, 0)[col_n].astype(int)

new_Freq_sel = pd.DataFrame(columns=col_n)
for i in col_n:
    new_Freq_sel[i] = Freq_sel[i].apply(lambda x: fun(x))

new_Freq_sel['Count'] = 1

After_Records_123 = new_Freq_sel.groupby(by=col_n, as_index=False).size().reset_index(name='Freq')


# %% Merged purposes (most significant for ANOVA after clustering):
# Cluster centroids: 1-6-8 ; 1-6 ; 5
# 替换小频率的choice，用lambda
def fun2(x):
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


Purpose_merged = Purpose_raw.copy()
Purpose_merged[col_n] = Purpose_merged[col_n].replace(np.nan, 0)[col_n].astype(int)

for i in col_n:
    Purpose_merged[i] = Purpose_merged[i].apply(lambda x: fun2(x))
Freq_purpose_merged = Purpose_merged[col_n].apply(pd.value_counts).replace(np.nan, 0).astype(int)
Freq_purpose_merged.rename(index={0: 'None'}, inplace=True)
# %%  4-19
# 做kmeans的frequency statistics，看聚类的效果
# kmeans_raw(merged)是与ENQ2006对应顺序的 purpose原始记录。 raw为未精简后的，merged为精简后的
# kmeans_raw = pd.read_excel('Purpose_statistics.xlsx', sheetname='kmeans_raw_set')
# kmeans_merged = pd.read_excel('Purpose_statistics.xlsx', sheetname='kmeans_merged_set')
#
## kmeans_raw.drop(kmeans_raw.columns[0],axis=1,inplace=True)
# kmeans_raw['Count'] = 1
# columns = ['観光目的①', '観光目的②', '観光目的③', 'cluster']
## frequency statistics
# k_r_123 = kmeans_raw.groupby(by = columns, as_index=False).size().reset_index(name='Freq')
#
# kmeans_merged['Count'] = 1
## frequency statistics
# k_m_123 = kmeans_merged.groupby(by = columns, as_index=False).size().reset_index(name='Freq')

# %% id-destination_set
# 4-25
Cluster_cor = pd.read_excel('./観光客の動向調査.xlsx', sheet_name='Correlation')
Cluster_cor.set_index(["ナンバリング"], inplace=True)

# Series既有数组的性质又有字典的性质

id_clst = Cluster_cor['Cluster']  # id和cluster对应的series

# cluster分别为：1,6,8 ; 1,6 ; 5

Dst_raw = OD_data.loc[:, ['ナンバリング', '出発地', '到着地']]

dst_res = pt.create_dst_res(Dst_raw)
# %%  Person - visited attraction areas
PT_data = pt.write_PT_num(Purpose_raw, dst_res)
# names of vistied places
PT_data2 = pt.write_PT_name(Purpose_raw, dst_res)
# %%  统计car travel dummy, transit dummy 和 Kyoto stay dummy

PT_dummy = pt.write_dummy(Purpose_raw, dst_res)
PT_dummy['Clusters'] = np.array(id_clst)
#  更改column顺序
cols = list(PT_dummy)
cols.insert(1, cols.pop(cols.index('Clusters')))
PT_dummy = PT_dummy.loc[:, cols]

#  写destination dummy
# destination frequency统计量
PT_dummy = pt.write_dest_dummy(PT_dummy)

means = PT_dummy.loc[:, '1':'59'].groupby([PT_dummy['Clusters'], PT_dummy['origin']]).mean()
counts = PT_dummy.loc[:, '1':'59'].groupby([PT_dummy['Clusters'], PT_dummy['origin']]).count()

means_origin = PT_dummy.loc[:, '1':'59'].groupby(PT_dummy['origin']).mean()

#  读取根据destination choice的clustering results

Dest_Clustering = scio.loadmat('Dest_Clustering.mat')['Index']
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
Places = pd.read_excel('./観光客の動向調査.xlsx', sheetname='Place_code')
Place_names = dict(zip(Places.no, Places.name))  # places code 和 name的字典

# trips chains and travel times
Trips, Trip_len = pt.create_trips(OD_data, PT_dummy)
Times, Travel_times = pt.create_travel_time(OD_data, PT_dummy)

Modes = pt.create_mode_split(OD_data, PT_dummy)
Costs = pt.create_costs(OD_data, PT_dummy)

# 根据实际trip chain 添加car travel flag
car_flag = pt.define_car_trip(PT_dummy, Modes, Trips)

PT = pd.concat([PT_dummy, car_flag['car_trip']], axis=1)
cols = list(PT)
cols.insert(6, cols.pop(cols.index('car_trip')))
PT = PT.loc[:, cols]

# # 根据实际trip chain 添加sightseeing bus travel flag
# sb_flag is modified to reflect both car trips (2) and sightseeing bus trips (1)
sb_flag = pt.sightseeing_bus_trip(PT_dummy, Modes, Trips)

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

# %% 统计每个cluster的destinaiton frequency
# 新建存放destination frequency的dataframe
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
pd_raw['ナンバリング'] = Purpose_raw['ナンバリング']
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


# %% 一行代码实现乘法口诀表
# print('\n'.join([' '.join ('%dx%d=%2d' % (x,y,x*y)  for x in range(1,y+1)) for y in range(1,10)]))

# %% 5-8 reduce and merge more purposes
# new purposes: (with respect to previously merged purposes)
#    1. (1,4) 名所・寺社見学・祭り・イベント　　ーー＞　cultural and art
#    2. (3) ナイトスポット
#    3. (2,5)　散策・体験活動
#    4. (6,7)　紅葉・自然
#    5. (8)　グルメ
#    6. (9)　買い物（お土産）

# classification function
def fun3(x):
    set1 = [1, 4]
    set2 = [3]
    set3 = [2, 5, 6, 7, 10, 12] + list(range(14, 18))
    set4 = [8, 9]
    set5 = [11]
    set6 = [13]
    if x in set1:
        return 1
    if x in set2:
        return 2
    if x in set3:
        return 3
    if x in set4:
        return 4
    if x in set5:
        return 5
    if x in set6:
        return 6
    else:
        return 0


Purpose_merged_more = Purpose_raw.copy()
Purpose_merged_more[col_n] = Purpose_merged_more[col_n].replace(np.nan, 0)[col_n].astype(int)

for i in col_n:
    Purpose_merged_more[i] = Purpose_merged_more[i].apply(lambda x: fun3(x))
