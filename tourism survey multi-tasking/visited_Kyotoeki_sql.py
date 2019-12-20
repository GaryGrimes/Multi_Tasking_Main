#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:02:36 2019

@author: gary
"""

if __name__ == '__main__':
    print('haha')
    # %% check 去了京都站周边玩的人他们的exit种类，以及是否是最后一个景点
    # counts = PT[PT['29']==1].shape[0],  # 819
    Kyoto_syuhen = PT[PT['29'] == 1].loc[:, 'ナンバリング':'Kyoto stay dummy']
    for i in Kyoto_syuhen.index:
        Kyoto_syuhen.at[i, 'exit'] = Kyoto_syuhen.loc[i, 'Visited'][-1]
    foo = Kyoto_syuhen['exit'].value_counts()
    print('Exit frequency: \n')
    for x in foo.index:
        t = 99 if x == 59 else x
        print(Place_names[t], ': ', foo[x])
    #  是否最后一个景点？
    for i in Kyoto_syuhen.index:
        Kyoto_syuhen.at[i, 'End?'] = 1 if 29 in Trips[Kyoto_syuhen.loc[i, 'ナンバリング']][-1, :] else 0
    print('As end of trip?\n', Kyoto_syuhen['End?'].value_counts())

    #  住在京都的人的destination frequency和外地人的区别
