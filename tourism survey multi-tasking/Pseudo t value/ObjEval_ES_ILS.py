import numpy as np
import pickle
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt

from slvr.SolverUtility_ILS import SolverUtility
import multiprocessing as mp

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Agent(object):
    agent_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Agent.agent_count += 1


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


def print_path(path_to_print):
    print(list(np.array(path_to_print) + 1))


def penalty(particle):
    _answer = [-0.02, -0.01, 0.3, 0.1]
    diff = np.array(_answer) - np.array(particle)
    _penalty = np.exp(np.linalg.norm(diff))
    return _penalty


def penalty2score(*args):
    if args:
        _scores = (1 / np.array(args) * 10000) ** 20
        return _scores
    else:
        return []


def score2penalty(*args):
    if args:
        _penalty = 10000 / (np.array(args) ** (1 / 20))
        return _penalty
    else:
        return []


# set optimum
# B_star = [-1., -0.036, 1.002, 0.108]

B_star = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]

if __name__ == '__main__':

    # %% setting up nodes
    node_num = 37  # Number of attractions. Origin and destination are excluded.

    Intrinsic_utilities = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Intrinsic Utility.xlsx'),
                                        sheet_name='data')
    utility_matrix = []
    for _idx in range(Intrinsic_utilities.shape[0]):
        temp = np.around(list(Intrinsic_utilities.iloc[_idx, 1:4]), decimals=3)
        utility_matrix.append(temp)
    utility_matrix = np.array(utility_matrix)

    Dwell_time = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Dwell time array.xlsx'),
                               index_col=0)
    # replace missing values by average of all samples
    Dwell_time.loc[35, 'mean'] = Dwell_time['mean'][Dwell_time['mean'] != 5].mean()  # Attraction 35
    dwell_vector = np.array(Dwell_time['mean'])

    # %% edge property
    Edge_time_matrix = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'transit_time_update2.xlsx'), index_col=0)

    edge_time_matrix = np.array(Edge_time_matrix)

    # Edge travel time
    # need several iterations to make sure direct travel is shorter than any detour

    no_update, itr = 0, 0
    # print('Starting travel_time_check...')
    for _ in range(3):
        while not no_update:
            # print('Current iteration: {}'.format(itr + 1))
            no_update = 1
            for i in range(edge_time_matrix.shape[0] - 1):
                for j in range(i + 1, edge_time_matrix.shape[0]):
                    time = edge_time_matrix[i, j]
                    shortest_node, shortest_time = 0, time
                    for k in range(edge_time_matrix.shape[0]):
                        if edge_time_matrix[i, k] + edge_time_matrix[k, j] < shortest_time:
                            shortest_node, shortest_time = k, edge_time_matrix[i, k] + edge_time_matrix[k, j]
                    if shortest_time < time:
                        no_update = 0
                        # print('travel time error between {0} and {1}, \
                        # shortest path is {0}-{2}-{1}'.format(i, j, shortest_node))
                        edge_time_matrix[j, i] = edge_time_matrix[i, j] = shortest_time
            itr += 1
            if no_update:
                # print('Travel time update complete.\n')
                pass

    # Edge travel cost (fare)
    Edge_cost_matrix = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'wide_transit_fare_matrix.csv'),
        index_col=0)
    # Edge travel distance
    Edge_distance_matrix = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'driving_wide_distance_matrix.xlsx'),
        index_col=0)

    edge_cost_matrix = np.array(Edge_cost_matrix)
    # distance matrix for path penalty evaluation
    edge_distance_matrix = np.array(Edge_distance_matrix)  # distance between attraction areas

    #  check UtilMatrix等各个matrix的shape是否正确。与NodeNum相符合
    if len(utility_matrix) != node_num:
        raise ValueError('Utility matrix error.')
    if edge_time_matrix.shape[0] != edge_time_matrix.shape[1]:
        raise ValueError('Time matrix error.')
    if edge_cost_matrix.shape[0] != edge_cost_matrix.shape[1]:
        raise ValueError('Cost matrix error.')
    if len(dwell_vector) != node_num:
        raise ValueError('Dwell time array error.')
    # setting up behavior parameters
    phi = 0.1

    # %% load agents
    with open(os.path.join(os.path.dirname(__file__), 'Database', 'transit_user_database.pickle'), 'rb') as file:
        agent_database = pickle.load(file)

    print('Setting up solver.')
    core_process = mp.cpu_count()  # species size (each individual is our parameters here)
    # itv = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]  # [i, j, k, l] for i in _ for j in ....就可以生成一个完整的permutation的list了
    # indices = [[i, j, k, l] for i in itv for j in itv for k in itv for l in itv]

    # create initial values of parameters
    sway = 0.01

    '''Generate the values to evaluate'''
    possible_values = []

    '''generate near values for B*'''
    cnt = 0

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
                    print('Duplicate: {}: {} and {}: {}.'.format(i, possible_values[i], j, possible_values[j]))
                    duplicate_idx.append(j)
                    flag = 1
    print(flag)

    Population = [possible_values[i] for i in range(len(possible_values)) if i not in duplicate_idx]

    # calculate score and record of the 1st generation
    time, itr = 0, 0
    Population_penalties, Population_scores = [], []

    s = []
    while Population:
        itr += 1
        print('------ Evaluation start for iteration {} ------\n'.format(itr))

        s = Population[:core_process]
        # todo evaluation都放到这里面来
        # evaluate scores for the first generation
        """multiprocessing: 建一个queue存结果，然后join"""

        jobs = []
        penalty_queue = mp.Queue()  # queue, to save results for multi_processing

        # calculate evaluation time
        start_time = datetime.datetime.now()

        for _id, parameter in enumerate(s):
            print('Starting process {} in {}'.format(_id + 1, len(s)))

            ALPHA = list(parameter[:2])
            BETA = [5] + list(parameter[2:])
            data_input = {'alpha': ALPHA, 'beta': BETA,
                          'phi': phi,
                          'util_matrix': utility_matrix,
                          'time_matrix': edge_time_matrix,
                          'cost_matrix': edge_cost_matrix,
                          'dwell_matrix': dwell_vector,
                          'dist_matrix': edge_distance_matrix}

            # start process
            process = mp.Process(target=SolverUtility.solver, args=(penalty_queue, _id, node_num, agent_database),
                                 kwargs=data_input, name='P{}'.format(_id + 1))
            jobs.append(process)
            process.start()

        for j in jobs:
            # join process
            j.join()

        end_time = datetime.datetime.now()
        print('------ Evaluation time for iteration {} : {}s ------\n'.format(itr, (end_time - start_time).seconds))

        time += (end_time - start_time).seconds
        print('------ Total time passed: {} hh {} mm {} ss ------\n'.format(time // 3600,
                                                                            time % 3600 // 60,
                                                                            time % 60))
        # 从 queue里取值
        Para_penalties_tuples = []
        while True:
            if penalty_queue.empty():  # 如果队列空了，就退出循环
                break
            else:
                Para_penalties_tuples.append(penalty_queue.get())

        para_penalties = []
        for _i in range(len(s)):
            for _tuple in Para_penalties_tuples:
                if _i == _tuple[0]:
                    para_penalties.append(_tuple[1])
                    break

        scores = list(penalty2score(para_penalties)[0])  # functions returns ndarray

        # write generation record and scores
        Population_penalties.extend(para_penalties)
        Population_scores.extend(scores)

        # print evaluation scores
        print('Evaluation scores:')
        for i, _ in enumerate(scores):
            print(
                'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e' % (i + 1, s[i][0],
                                                                                            s[i][1],
                                                                                            s[i][2],
                                                                                            s[i][3],
                                                                                            _))

        del Population[:core_process]
        # todo 计算还剩下多少parameter需要计算，当前的进度（%)，预估时间

    # %% save results into DF
    # Population = list(possible_values)
    Population = [possible_values[i] for i in range(len(possible_values)) if i not in duplicate_idx]
    Res = pd.DataFrame(columns=['index', 'a1', 'a2', 'b2', 'b3', 'penalty', 'score'])
    Res['index'] = range(len(Population))
    Res.loc[:, 'a1':'b3'] = Population
    Res['score'] = Population_scores
    Res['penalty'] = Population_penalties

    Res.to_excel('Iteration result for t value evaluation.xlsx')
