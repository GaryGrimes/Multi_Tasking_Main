import numpy as np
import pickle
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from SolverUtility import SolverUtility
import multiprocessing as mp
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def selection(s_size, _scores):
    insertion_size = 3
    best_one_idx = np.argsort(_scores)[-1]
    f_sum = sum(_scores)
    prob = [_ / f_sum for _ in _scores]
    # calculate accumulated prob
    prob_acu = [sum(prob[:_]) + prob[_] for _ in range(len(prob))]
    prob_acu[-1] = 1

    # 画柱状堆积图
    # TODO 改 堆积图 bottom 相当于堆积
    # 用queue，while SCORES: 每次pop一个，另一个是bottom

    _idx, _bottom = 0, 0
    plt.figure(dpi=100)
    while prob:
        if not _idx:
            plt.bar(1, prob[0], label="parameter {}".format(_idx))
        else:
            plt.bar(1, prob[0], bottom=_bottom, label="parameter {}".format(_idx))
        _bottom += prob.pop(0)
        _idx += 1

    # for _idx, _prob in enumerate(prob):
    #     if _idx == 0:
    #         plt.bar(_prob, align="center", color="#66c2a5", label="parameter {}".format(_idx + 1))
    #     else:
    #     _bottom = _prob

    plt.legend()

    plt.show()
    # 柱状堆积图

    # return selected idx
    indices = []
    for _ in range(s_size - insertion_size):
        random_num = np.random.rand()
        indices.append(next(_x[0] for _x in enumerate(prob_acu) if _x[1] > random_num))  # x is a tuple
    # insert best results from history
    indices.extend(insertion_size * [best_one_idx])
    return indices


if __name__ == '__main__':
    SCORES = [1.6260083692733374e-05,
              1.3117795092636346e-05,
              1.0762469951737046e-05,
              1.3117795092636346e-05,
              1.5047574020949666e-05,
              1.4459921673313616e-05,
              1.6260083692733374e-05,
              1.5047574020949666e-05,
              1.5047574020949666e-05,
              1.0762469951737046e-05,
              1.3117795092636346e-05,
              1.5047574020949666e-05,
              1.6527748771695515e-05,
              1.6527748771695515e-05,
              1.6527748771695515e-05]
    print(selection(15, SCORES))
