import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import os
import pylab as pl
import itertools

'''  to compute all possible permutations
# using itertools.product()'''


def get_score(parameter, lib, res):
    key = tuple(parameter)
    return res[lib.index(key)]


def f(beta):
    global Population, pnt
    # todo 两个变量parameter和res，分别记录待计算的参数和已计算的值
    _idx = Population.index(beta)
    return res[_idx]


def f_1(beta, pos, sway):  # *loc 输入的是i,j i.e. 在哪两个方向上变化
    _center = list(beta)
    _epsilon = abs(sway * np.array(beta))  # todo 正式的code里epsilon是变化的，按array来定

    _left, _right = list(_center), list(_center)
    _left[pos], _right[pos] = _left[pos] - _epsilon[pos], _right[pos] + _epsilon[pos]

    _l_res = (f(_left) - f(beta)) / -_epsilon[pos]
    _r_res = (f(_right) - f(beta)) / _epsilon[pos]
    return _l_res, _r_res


def f_2(beta, loc, sway):
    d_1, d_2 = loc[0], loc[1]
    _epsilon = abs(sway * np.array(beta))

    _left = list(beta)  # 仅取前差分算
    _left[d_2] -= _epsilon[d_2]
    return (f_1(_left, d_1, sway)[0] - f_1(beta, d_1, sway)[0]) / -_epsilon[d_2]  # 取前差分算


if __name__ == '__main__':
    # set print out decimals
    np.set_printoptions(precision=3)

    # * ---------------- Read Evaluation results ---------------- *
    # read from evaluation results. Parameter evaluation result

    evaluation_result = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Iteration result for t value evaluation 0.1.xlsx'), index_col=1)

    pnt = list(-evaluation_result.penalty)

    # * ---------------- Generate the values to evaluate ---------------- *

    sway = 0.001

    '''Generate the values to evaluate'''
    possible_values = []

    '''generate near values for B*'''
    cnt = 0
    # set optimum
    B_star = [-1., -0.036, 1.002, 0.108]

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

    # get epsilon
    epsilon = abs(np.array(B_star) * sway)  # 让epsilon为正

    # Parameters = list(itertools.product(*possible_values))  # ‘*’ 操作符的作用是将元组“解包”

    # Res = np.random.randn(len(possible_values))  # only for test

    # ---------------------------- Test/ Value check ------------------------- #

    # calculate gradient
    Gradient = []

    # # check the gradient for beta[1]
    # print('\n')
    # print('Gradient check at alpha2: \n')
    # test_left = [-1.0, -0.036359999999999996, 1.002, 0.108]
    # print('Parameter score _left: {}, Beta*: {}'.format(f(test_left), f(B_star)))
    # test_right = [-1.0, -0.03564, 1.002, 0.108]
    # print('Parameter score _right: {}, Beta*: {} \n'.format(f(test_right), f(B_star)))
    #
    # # check the gradient for beta[3]
    # print('Gradient check at beta3: \n')
    # test_left = [-1.0, -0.036, 1.002, 0.10692]
    # print('Parameter score _left: {}, Beta*: {}'.format(f(test_left), f(B_star)))
    # test_right = [-1.0, -0.036, 1.002, 0.10908]
    # print('Parameter score _right: {}, Beta*: {}'.format(f(test_right), f(B_star)))

    test = f_1(B_star, 3, sway)

    # ---------------------------- Calculation ------------------------- #

    # gradient
    for i in range(len(B_star)):
        Gradient.append(f_1(B_star, i, sway))

    # calculate second derivative

    SecondDerivative = []
    # second gradient
    for i in range(len(B_star)):
        temp = []
        for j in range(len(B_star)):
            temp.append(f_2(B_star, [i, j], sway))
        SecondDerivative.append(temp)

    Gradient = np.array(Gradient).round(3)
    SecondDerivative = np.array(SecondDerivative).round()

    # ---------------------------- calculate pesudo t value  ------------------------- #

    variance = np.linalg.inv(-SecondDerivative)
    std_err = np.sqrt(np.diag(variance))

    # ---------------------------- print results ------------------------- #

    print('The numerical gradient matrix: \n {}\n'.format(Gradient))

    # print second derivatives
    print('The second derivative matrix:\n {}'.format(SecondDerivative))

    # ---------------------------- reference ------------------------- #
'''    for i in range(len(B_star)):
        center = list(B_star)
        # TODO calculate derivative for each entry, and add it to the 'Derivative' vector
        left_point, right_point = list(center), list(center)
        # get left, right points with little epsilon
        left_point[i], right_point[i] = left_point[i] - epsilon[i], right_point[i] + epsilon[i]
        # find indices
        left_point, right_point, center = tuple(left_point), tuple(right_point), tuple(center)
        left_gradient = - (Res[possible_values.index(tuple(left_point))] - Res[possible_values.index(tuple(center))]) / \
                        epsilon[i]  # left - center / -e
        right_gradient = (Res[possible_values.index(tuple(right_point))] - Res[possible_values.index(tuple(center))]) / \
                         epsilon[i]  # right - center / e
        Derivative.append({'left': left_gradient, 'right': right_gradient})

    # calculate second derivatives
    Hessian = np.zeros([len(B_star), len(B_star)])'''

'''    res = [[i, j, k] for i in list1
           for j in list2
           for k in list3] '''
