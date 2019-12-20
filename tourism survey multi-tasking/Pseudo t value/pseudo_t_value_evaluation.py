import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import os
import ObjEval_ES_ILS as obj
import pylab as pl
import itertools

'''  to compute all possible permutations
# using itertools.product()'''


def get_score(parameter, lib, res):
    key = tuple(parameter)
    return res[lib.index(key)]


def f(beta):
    global Population, pnt, score
    # todo 两个变量parameter和res，分别记录待计算的参数和已计算的值
    _idx = Population.index(beta)
    return pnt[_idx]


def f_1(beta, pos):  # *loc 输入的是i,j i.e. 在哪两个方向上变化
    _center = list(beta)
    _epsilon = abs(0.01 * np.array(beta))  # todo 正式的code里epsilon是变化的，按array来定
    for _ in range(len(_epsilon)):
        _epsilon[_] = 0.0001 if _epsilon[_] < 0.0001 else _epsilon[_]

    _left, _right = list(_center), list(_center)

    _left[pos], _right[pos] = _left[pos] - _epsilon[pos], _right[pos] + _epsilon[pos]

    _l_res = (f(_left) - f(beta)) / -_epsilon[pos]
    _r_res = (f(_right) - f(beta)) / _epsilon[pos]
    return _l_res, _r_res


def f_2(beta, loc):
    d_1, d_2 = loc[0], loc[1]
    _epsilon = abs(0.01 * np.array(beta))
    for _ in range(len(_epsilon)):
        _epsilon[_] = 0.0001 if _epsilon[_] < 0.0001 else _epsilon[_]
    _left = list(beta)  # 仅取前差分算
    _left[d_2] -= _epsilon[d_2]
    return (f_1(_left, d_1)[0] - f_1(beta, d_1)[0]) / -_epsilon[d_2]  # 取前差分算


if __name__ == '__main__':
    # set print out decimals
    # np.set_printoptions(precision=3)

    # * ---------------- Read Evaluation results ---------------- *
    # read from evaluation results. Parameter evaluation result

    evaluation_result = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'objective value evaluation', 'Dec5',
                     'Iteration result for t value evaluation.xlsx'),
        index_col=1)

    pnt = list(-evaluation_result.penalty)
    score = list(evaluation_result.score)

    # * ---------------- Generate the values to evaluate ---------------- *

    sway = 0.01

    '''Generate the values to evaluate'''
    possible_values = []

    '''generate near values for B*'''
    cnt = 0
    # set optimum
    # B_star = [-1., -0.036, 1.002, 0.108]

    B_star = obj.B_star

    # get epsilon
    epsilon = abs(np.array(B_star) * sway)  # 让epsilon为正
    for i in range(len(B_star)):
        _ = np.array([0, 0, 0, 0])
        _[i] = 1
        possible_values.append(list(np.array(B_star) + epsilon * _))
        print('No. {}: {}, modified at position {}, + '.format(cnt, list(np.array(B_star) + epsilon * _), i))
        cnt += 1

        possible_values.append(list(np.array(B_star) - epsilon * _))
        print('No. {}: {}, modified at position {}, - '.format(cnt, list(np.array(B_star) - epsilon * _), i))
        cnt += 1

    # second derivative会用到的near values
    for i in range(len(B_star)):
        Beta = list(B_star)
        # set 'optimum' (即center value).二次导的时候递归计算会用到的value.
        Beta[i] -= epsilon[i]  # 二次导仅使用前差分算
        print('\nCurrent beta: {}\n'.format(Beta))
        # get epsilon
        _epsilon = abs(np.array(Beta) * sway)  # 让epsilon为正
        for j in range(len(Beta)):
            _ = np.array([0, 0, 0, 0])
            _[j] = 1
            possible_values.append(list(np.array(Beta) + _epsilon * _))
            print('No. {}: {}, modified at position {}, + '.format(cnt, list(np.array(Beta) + _epsilon * _), j))
            cnt += 1
            possible_values.append(list(np.array(Beta) - _epsilon * _))
            print('No. {}: {}, modified at position {}, - '.format(cnt, list(np.array(Beta) - _epsilon * _), j))
            cnt += 1

    # 最后加上B* 本身
    possible_values.append(list(np.array(B_star)))

    flag, duplicate_idx = 0, []
    for i in range(len(possible_values)):
        for j in range(len(possible_values)):
            if j > i:
                if possible_values[i] == possible_values[j]:
                    # print('Duplicate: {}: {} and {}: {}.'.format(i, possible_values[i], j, possible_values[j]))
                    duplicate_idx.append(j)
                    flag = 1

    Population = [possible_values[i] for i in range(len(possible_values)) if i not in duplicate_idx]

    # ---------------------------- Setup ------------------------- #

    # set parameters
    sway = 0.01

    # set optimum

    B_star = obj.B_star

    # get epsilon
    epsilon = abs(np.array(B_star) * sway)  # 让epsilon为正

    # calculate gradient
    Gradient = []

    # ---------------------------- Calculation ------------------------- #

    # gradient
    for i in range(len(B_star)):
        Gradient.append(f_1(B_star, i))

    # calculate second derivative

    SecondDerivative = []
    # second gradient
    for i in range(len(B_star)):
        temp = []
        for j in range(len(B_star)):
            temp.append(f_2(B_star, [i, j]))
        SecondDerivative.append(temp)

    Gradient = np.array(Gradient)
    SecondDerivative = np.array(SecondDerivative)

    # ---------------------------- calculate pesudo t value  ------------------------- #

    variance = np.linalg.inv(SecondDerivative)
    std_err = np.sqrt(np.diag(variance))

    # ---------------------------- print results ------------------------- #

    print('The numerical gradient matrix: \n {}\n'.format(Gradient))

    # print second derivatives
    print('The Hessian matrix:\n {}'.format(SecondDerivative))

    # print variance matrix
    print('The variance matrix:\n {}'.format(variance))