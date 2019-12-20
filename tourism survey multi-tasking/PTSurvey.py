#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:16:26 2019

@author: gary
"""
import pandas as pd
import numpy as np


class PTSurvey(object):

    def create_dst_res(Dst_raw):
        dst_res = {}  # dict
        k = 0
        while k < len(Dst_raw):
            places = []
            key = Dst_raw.loc[k, 'ナンバリング']
            while k + 1 < len(Dst_raw) and Dst_raw.loc[k + 1, 'ナンバリング'] == key:
                places.append(Dst_raw.loc[k, '出発地'])
                k += 1
            places.extend([Dst_raw.loc[k, '出発地'], Dst_raw.loc[k, '到着地']])
            dst_res[key] = places
            k += 1
        return dst_res

    def write_PT_name(Purpose_raw, dst_res):
        Places = pd.read_excel('./観光客の動向調査.xlsx', sheetname='Place_code')
        # 把places转为字符串，然后将dict转为df就好了
        PT_data = pd.DataFrame(columns=['ナンバリング', 'Places'])
        PT_data['ナンバリング'] = Purpose_raw['ナンバリング']
        for i in range(len(PT_data)):
            key = PT_data.loc[i, 'ナンバリング']
            try:
                places = set([59 if x == 99 else x for x in dst_res[key]])
                places_name = ', '.join([Places.loc[k - 1, 'name'] for k in places])
            except:
                places_name = 'None'
            PT_data.at[i, 'Places'] = places_name
        # PT_data.to_csv("PT_data.csv",index=False,sep=',')
        return PT_data

    def write_PT_num(Purpose_raw, dst_res):
        def is_attration(x):
            return 1 if x in list(range(1, 38)) else 0
            # 把places转为字符串，然后将dict转为df就好了

        PT_data = pd.DataFrame(columns=['ナンバリング', 'Places'])
        PT_data['ナンバリング'] = Purpose_raw['ナンバリング']
        for i in range(len(PT_data)):
            key = PT_data.loc[i, 'ナンバリング']
            try:
                places = set([59 if x == 99 else x for x in dst_res[key]])
                places = list(filter(is_attration, places))
            except:
                pass
            PT_data.at[i, 'Places'] = places
        # PT_data.to_csv("PT_data.csv",index=False,sep=',')
        return PT_data

    def write_dummy(Purpose_raw, dst_res):
        import numpy as np
        data = pd.DataFrame(columns=['ナンバリング', 'origin', 'Visited',
                                     'car dummy', 'transit dummy', 'Kyoto stay dummy'])
        data['ナンバリング'] = Purpose_raw['ナンバリング']
        for i in range(len(data)):
            key = data.loc[i, 'ナンバリング']
            origin = car_dummy = transit_dummy = Kyoto_stay_dummy = np.nan
            try:
                places = [59 if x == 99 else x for x in dst_res[key]]
                origin = places[0]
                Kyoto_stay_dummy, transit_dummy, car_dummy = [
                    origin in range(38, 40), origin in range(40, 48), origin in range(48, 58)]
                if Kyoto_stay_dummy:
                    origin = 1
                elif transit_dummy:
                    origin = 2
                else:
                    origin = 3
            except:
                places = np.nan
            data.loc[i, 'origin'] = origin
            data.loc[i, 'Visited':'Kyoto stay dummy'] = [places, car_dummy, transit_dummy, Kyoto_stay_dummy]
        return data

    def write_dest_dummy(PT_dummy):
        import numpy as np
        coln = [str(x) for x in range(1, 60)]
        data = pd.DataFrame(columns=coln)
        for i in range(len(PT_dummy)):
            try:
                places_dummy = np.zeros(59)
                for x in PT_dummy.loc[i, 'Visited']:
                    places_dummy[x - 1] = 1
            except:
                places_dummy = np.nan
            data.loc[i, coln] = places_dummy
        PT_dummy = pd.concat([PT_dummy, data], axis=1)
        return PT_dummy

    def create_TP_DataBase(Data, PT_dummy, Dst_raw):  # Data: sheet 'OD'
        import numpy as np
        PT_DB = pd.DataFrame(columns=['ナンバリング', 'trip_len', 'origin', 'destination',
                                      'trips', 'time', 'modes', 'food', 'souvenir', 'dissatisfaction'])
        PT_DB['ナンバリング'] = PT_dummy['ナンバリング']
        k = 0
        while k < len(Data):
            start = k
            key = Data.loc[k, 'ナンバリング']
            while k + 1 < len(Dst_raw) and Dst_raw.loc[k + 1, 'ナンバリング'] == key:
                k += 1
            end = k

            trip_len = end - start + 1
            trips = np.array(Data.loc[start:end, '出発地':'到着地'])
            origin, destination = trips[0, 0], trips[-1, -1]
            idx = PT_DB[PT_DB['ナンバリング'] == key].index.tolist()[0]  # 问题出在index上
            # index 返回的是一个数组，所以赋值的时候也要嵌套括号
            PT_DB.loc[idx, 'trip_len'], PT_DB.loc[idx, 'origin'], PT_DB.loc[
                idx, 'destination'] = trip_len, origin, destination
            PT_DB.at[idx, 'trips'] = trips
            PT_DB.at[idx, 'time'] = np.array(Data.loc[start:end, '出発時':'到着分'])
            PT_DB.at[idx, 'modes'] = np.array(Data.loc[start:end, '交通手段１':'交通手段６'])
            PT_DB.at[idx, 'food'] = np.array(Data.loc[start:end, '飲食費'])
            PT_DB.at[idx, 'souvenir'] = np.array(Data.loc[start:end, '土産代'])
            PT_DB.at[idx, 'dissatisfaction'] = np.array(Data.loc[start:end, '不満点１':'不満点６'])
            k += 1
        return PT_DB

    def create_trips(Data, PT_dummy):
        import numpy as np
        Trips = {}
        Trip_len = {}
        keys = list(PT_dummy['ナンバリング'])
        for i in keys:
            try:
                trips = np.array(Data.loc[Data[Data['ナンバリング'] == i].
                                 index.tolist(), '出発地':'到着地'].replace(np.nan, 0))
            except:
                trips = np.nan
            Trips[i] = trips
            Trip_len[i] = len(trips)
        return Trips, Trip_len

    def create_travel_time(Data, PT_dummy):
        import numpy as np
        Times = {}
        Travel_time = {}
        keys = list(PT_dummy['ナンバリング'])
        for i in keys:
            try:
                times = np.array(Data.loc[Data[Data['ナンバリング'] == i].
                                 index.tolist(), '出発時':'到着分'].replace(np.nan, 0))
                t_times = np.array(
                    [x if x > 0 else 0 for x in 60 * (times[:, 2] - times[:, 0]) + (times[:, 3] - times[:, 1])])
            except:
                times = np.nan
                t_times = np.nan
            Times[i] = times
            Travel_time[i] = t_times
        return Times, Travel_time

    def create_mode_split(Data, PT_dummy):
        Modes = {}
        keys = list(PT_dummy['ナンバリング'])
        for i in keys:
            try:
                modes = np.array(Data.loc[Data[Data['ナンバリング'] == i].
                                 index.tolist(), '交通手段１':'交通手段６'].replace(np.nan, 0))
            except:
                modes = np.nan
            Modes[i] = modes
        return Modes

    def create_costs(Data, PT_dummy):
        Costs = {}
        keys = list(PT_dummy['ナンバリング'])
        for i in keys:
            try:
                costs = np.array(Data.loc[Data[Data['ナンバリング'] == i].
                                 index.tolist(), ['飲食費', '土産代']].replace(np.nan, 0))
            except:
                costs = np.nan
            Costs[i] = costs
        return Costs

    # 需要 Modes, Trips作为输入变量
    def define_car_trip(PT_dummy, Modes, Trips):
        is_car_trip = pd.DataFrame(columns=['ナンバリング', 'car_trip'])
        is_car_trip['ナンバリング'] = PT_dummy['ナンバリング']
        for i in range(len(PT_dummy)):
            car_flag = 0
            key = is_car_trip.loc[i, 'ナンバリング']
            try:
                start, end = 0, len(Trips[key]) - 1
                while Trips[key][start, 1] not in range(1, 38):
                    start += 1
                while Trips[key][end, 0] not in range(1, 38):
                    end -= 1
                start += 1  # 从第二个dest开始算起。people might park their cars and transfer
                if start <= end:
                    if 5 in Modes[key][start:end, :] or 7 in Modes[key][start:end, :]:
                        car_flag = 1
            except:
                pass
            is_car_trip.loc[i, 'car_trip'] = car_flag
        return is_car_trip

    def sightseeing_bus_trip(PT_dummy, Modes, Trips):
        is_sb_trip = pd.DataFrame(columns=['ナンバリング', 'sb_trip'])
        is_sb_trip['ナンバリング'] = PT_dummy['ナンバリング']
        for i in range(len(PT_dummy)):
            sb_flag = 0
            key = is_sb_trip.loc[i, 'ナンバリング']
            try:
                start, end = 0, len(Trips[key]) - 1
                while Trips[key][start, 1] not in range(1, 38):
                    start += 1
                while Trips[key][end, 0] not in range(1, 38):
                    end -= 1
                if start <= end:
                    if 5 in Modes[key][start:end, :] or 7 in Modes[key][start:end, :]:
                        sb_flag = 1
                    elif 8 in Modes[key][start:end, :] or 9 in Modes[key][start:end, :]:
                        sb_flag = 2
                    else:
                        pass
            except:
                pass
            is_sb_trip.loc[i, 'sb_trip'] = sb_flag
        return is_sb_trip

    def create_PT(Purpose_raw, Dst_raw):
        import pandas as pd
        PT_data = pd.DataFrame(columns=['ナンバリング', 'Places'])
        PT_data['ナンバリング'] = Purpose_raw['ナンバリング']
        key = 0
        Places = []
        destinations = []
        for k in range(len(Dst_raw)):
            flag = 1 if key == Dst_raw.loc[k, 'ナンバリング'] else 0
            if flag == 0 and k:
                Places.append()
                destinations = []
            key = Dst_raw.loc[k, 'ナンバリング']
            destinations.append(Dst_raw.loc[k, '出発地']);
            destinations.append(Dst_raw.loc[k, '到着地'])
        #    PT_data['Places'] = Places[1::]
        return Places

    def Trip_n(key, Place_names, Trips):
        trips = Trips[key].tolist()
        visited = []
        while trips:
            places = trips.pop(0)
            visited.append(list(map(lambda x: Place_names[x], places)))
        return visited
