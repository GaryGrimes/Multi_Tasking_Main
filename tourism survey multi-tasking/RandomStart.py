import numpy as np
import pickle
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt

from SolverUtility import SolverUtility
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


if __name__ == '__main__':
    # %% Solver Setup
    NodeNum = 37  # number of attractions. Origin and destination are excluded.

    Intrinsic_utilities = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Intrinsic Utility.xlsx'),
                                        sheet_name='data')
    # node property
    UtilMatrix = []
    for _idx in range(Intrinsic_utilities.shape[0]):
        temp = np.around(list(Intrinsic_utilities.iloc[_idx, 1:4]), decimals=3)
        UtilMatrix.append(temp)

    Dwell_time = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Dwell time array.xlsx'),
                               index_col=0)
    # check and clean average dwell time
    Dwell_time.loc[35, 'mean'] = Dwell_time['mean'][Dwell_time['mean'] != 5].mean()
    DwellArray = np.array(Dwell_time['mean'])

    # edge property
    Edge_time_matrix = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'transit_time_update2.xlsx'), index_col=0)
    # %% need several iterations to make sure direct travel is shorter than any detour
    NoUpdate, itr = 0, 0
    for _ in range(3):
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
                        print('travel time error between {0} and {1}, shortest path is {0}-{2}-{1}'.format(i, j,
                                                                                                           shortest_node))
                        Edge_time_matrix.loc[j, i] = Edge_time_matrix.loc[i, j] = shortest_time
            itr += 1
            if NoUpdate:
                print('Travel time update complete.\n')
    Edge_cost_matrix = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'wide_transit_fare_matrix.csv'),
        index_col=0)
    Edge_distance_matrix = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'driving_wide_distance_matrix.xlsx'),
        index_col=0)

    TimeMatrix, CostMatrix = np.array(Edge_time_matrix), np.array(Edge_cost_matrix)  # time in min

    # distance matrix for path penalty
    DistMatrix = np.array(Edge_distance_matrix)  # distance between attraction areas

    #  check UtilMatrix等各个matrix的shape是否正确。与NodeNum相符合
    if len(UtilMatrix) != NodeNum:
        raise ValueError('Utility matrix error.')
    if TimeMatrix.shape[0] != TimeMatrix.shape[1]:
        raise ValueError('Time matrix error.')
    if CostMatrix.shape[0] != CostMatrix.shape[1]:
        raise ValueError('Cost matrix error.')
    if len(DwellArray) != NodeNum:
        raise ValueError('Dwell time array error.')

    # %% load agents
    with open(os.path.join(os.path.dirname(__file__), 'Database', 'agent_database.pickle'), 'rb') as file:
        agent_database = pickle.load(file)

    print('Setting up solver.')
    # parameter setup
    phi = 0.1

    core_process = 16  # species size (each individual is our parameters here)
    itv = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    size_itv = len(itv)
    # create parameter columns
    """ 不用下面这么麻烦啦
    beta4_col = [0.01, 0.03, 0.1, 0.3, 1, 3, 10] * size_itv ** 3
    beta3_col = ([0.01] * 7 + [0.03] * 7 + [0.1] * 7 + [0.3] * 7 + [1] * 7 + [3] * 7 + [10] * 7) * 7 ** 2
    beta2_col = ([0.01] * 7 ** 2 + [0.03] * 7 ** 2 + [0.1] * 7 ** 2 + [0.3] * 7 ** 2 + [1] * 7 ** 2 + [3] * 7 ** 2 + [
        10] * 7 ** 2) * 7
    beta1_col = ([0.01] * 7 ** 3 + [0.03] * 7 ** 3 + [0.1] * 7 ** 3 + [0.3] * 7 ** 3 + [1] * 7 ** 3 + [3] * 7 ** 3 + [
        10] * 7 ** 3) * 7
    """
    # create initial values of parameters
    # 聪明的操作
    indices = [[(i // size_itv ** 3) % 7, (i // size_itv ** 2) % 7, (i // size_itv) % 7, i % 7] for i in
               range(size_itv ** 4)]
    s = []

    Population = [[itv[indices[j][i]] for i in range(4)] for j in range(len(indices))]
    # manually insert reasonable parameters

    # alpha should have negative values
    for _ in Population:
        for j in range(len(_)):
            if j < 2:
                _[j] = -_[j]

    # calculate score and record of the 1st generation
    time, itr = 0, 0
    Population_penalties, Population_scores = [], []
    while Population:
        itr += 1
        print('------ Evaluation start for iteration {} ------\n'.format(itr))

        s = Population[:core_process]
        # todo evaluation都放到这里面来
        # evaluate scores for the first generation
        """multiprocessing: 建一个queue存结果，然后join"""

        jobs = []
        PENALTIES = mp.Queue()  # queue, to save results for multi_processing

        # calculate evaluation time
        start_time = datetime.datetime.now()

        for _id, parameter in enumerate(s):
            print('Starting process {} in {}'.format(_id + 1, len(s)))

            ALPHA = parameter[:2]
            BETA = [5] + parameter[2:]
            arg_input = {'alpha': ALPHA, 'beta': BETA, 'phi': phi,
                         'util_matrix': UtilMatrix, 'time_matrix': TimeMatrix, 'cost_matrix': CostMatrix,
                         'dwell_matrix': DwellArray, 'dist_matrix': DistMatrix}
            # start process
            process = mp.Process(target=SolverUtility.solver, args=(PENALTIES, _id, NodeNum, agent_database),
                                 kwargs=arg_input, name='P{}'.format(_id + 1))
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
            if PENALTIES.empty():  # 如果队列空了，就退出循环
                break
            else:
                Para_penalties_tuples.append(PENALTIES.get())

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


    # # %% save results into DF
    # Res = pd.DataFrame(columns=['itr', 'a1', 'a2', 'b2', 'b3', 'penalty', 'score', 'record_penalty', 'record'])
    # Res['itr'] = range(itr_max)
    # Res.loc[:, 'a1':'b3'] = x_max
    # Res['score'] = gnr_max
    # Res['penalty'] = score2penalty(gnr_max)[0]
    # Res['record_penalty'] = score2penalty(y_max)[0]
    # Res['record'] = y_max
    # Res.to_excel('Iteration result.xlsx')
    #
    # # save parameters into DF
    # df_parameter = pd.DataFrame.from_dict(PARAMETER, orient='index')
    # df_parameter.to_excel('Parameter in each iteration.xlsx')
