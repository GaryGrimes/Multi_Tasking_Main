'''Modified on Oct. 15'''

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


def selection(s_size, _scores):
    insertion_size = 3
    best_one_idx = np.argsort(_scores)[-1]
    f_sum = sum(_scores)
    prob = [_ / f_sum for _ in _scores]
    # calculate accumulated prob
    prob_acu = [sum(prob[:_]) + prob[_] for _ in range(len(prob))]
    prob_acu[-1] = 1

    # return selected idx
    indices = []
    for _ in range(s_size - insertion_size):
        random_num = np.random.rand()
        indices.append(next(_x[0] for _x in enumerate(prob_acu) if _x[1] > random_num))  # x is a tuple
    # insert best results from history
    indices.extend(insertion_size * [best_one_idx])
    return indices


def mutation(prob, best_score, population, population_scores):
    insertion_size = round(len(population) / 3)
    learn_rate = [0.01, 0.01, 0.01, 0.02]
    species = []
    best = list(population[np.argsort(population_scores)[-1]])

    # pick the largest 5 individuals to perform
    for _index, _i in enumerate(population):
        mut_temp = np.random.rand()
        if mut_temp < prob:  # perform mutation, else pass
            _score = population_scores[_index]
            weight = 4 * (np.abs(_score - best_score) / best_score)  # 0 <= weight < 5
            _new_individual = []
            # alphas should < 0
            for _j, _par_a in enumerate(_i[:2]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par_a + _gain > 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])
                _par_a += _gain  # update parameter
                _new_individual.append(_par_a)
            # betas should >= 0
            for _k, _par_b in enumerate(_i[2:]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _k])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par_b + _gain < 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _k])
                _par_b += _gain  # update parameter
                _new_individual.append(_par_b)
            species.append(_new_individual)
        else:
            species.append(_i)
    # insert the best solution so far
    """ always preserve the best solution """
    species.extend(insertion_size * [best])
    return species


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
    # replace missing values by average of all samples
    Dwell_time.loc[35, 'mean'] = Dwell_time['mean'][Dwell_time['mean'] != 5].mean()  # Attraction 35
    DwellArray = np.array(Dwell_time['mean'])

    # edge property
    Edge_time_matrix = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'transit_time_update2.xlsx'), index_col=0)
    # %% Edge properties
    # Edge travel time
    # need several iterations to make sure direct travel is shorter than any detour
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
    # Edge travel cost (fare)
    Edge_cost_matrix = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'wide_transit_fare_matrix.csv'),
        index_col=0)
    # Edge travel distance
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
    # utility parameter
    phi = 0.1
    # parameter setup
    inn = 12  # species size (each individual in current generation is a vector of behavioral parameters)
    itr_max = 200
    prob_mut = 1  # parameter mutation probability
    """ memo is to store parameter --> Path similarity objective function"""

    memo_parameter, memo_penalty = [], []  # memo stores parameters from last 2 iterations
    PARAMETER = {}

    # s = []  # species set of parameters

    # test s with fixed input
    s = [
        [-0.37974139765941783,
         -0.2558939179688775,
         0.04893578316589828,
         0.05210324618171788],
        [-0.08918215758059533,
         -0.24405051342668937,
         0.027020415460782002,
         0.2056606245278888],
        [-0.16167280374767834,
         -0.2411843920976503,
         0.03404410015008346,
         0.3076044553748146],
        [-0.46797554664851887,
         -0.08691471688216373,
         0.27465618122012814,
         0.8535210297561443],
        [-0.16654700848822268,
         -0.0887516253882134,
         0.14708878950043483,
         0.3303207960587167],
        [-0.3236607278310718,
         -0.0668914251165349,
         0.19367692132502703,
         0.4580954274520535],
        [-0.3232875461086227,
         -0.028247387947372693,
         0.24030778479735282,
         0.3322569213448343],
        [-0.059277941835195136,
         -0.4592104661803278,
         0.241806890829197,
         0.43319110214340956]
    ]

    # generate first population
    """ beta 1 is fixed here"""

    for i in range(inn - len(s)):  # 补全s
        # random alphas
        a1, a2 = np.random.uniform(-0.5, -0.01), np.random.uniform(-0.5, -0.01)
        # random betas
        b2, b3 = np.random.uniform(0.01, 0.3), np.random.uniform(0.02, 1)
        s.append([a1, a2, b2, b3])
    # manually insert reasonable parameters
    s[-2], s[-1] = [-0.05, -0.05, 0.03, 0.1], [-0.03, -0.01, 0.02, 0.1]

    y_mean, y_max, x_max, gnr_max = [], [], [], []  # 记录平均score, 每一世代max score， 每世代最佳个体

    # calculate score and record of the 1st generation
    print('------ Initialization: first generation ------')

    # evaluate scores for the first generation
    """multiprocessing: 建一个queue存结果，然后join"""
    jobs = []
    PenaltyQueue = mp.Queue()  # queue, to save results for multi_processing
    # PENALTIES = {}  # queue, to save results for multi_processing

    # calculate evaluation time
    start_time = datetime.datetime.now()

    for idx, parameter in enumerate(s):
        print('Starting process {} in {}'.format(idx + 1, inn))

        # check existence of parameter in memory
        if parameter in memo_parameter:
            PenaltyQueue.put((idx, memo_penalty[memo_parameter.index(parameter)]))
            print('The {}th parameter is sent from history (with index {}), with score: {}'.format(idx,
                                                                                                   memo_parameter.index(
                                                                                                       parameter),
                                                                                                   memo_penalty[
                                                                                                       memo_parameter.index(
                                                                                                           parameter)]))
        else:
            ALPHA = parameter[:2]
            BETA = [5] + parameter[2:]
            arg_input = {'alpha': ALPHA, 'beta': BETA, 'phi': phi,
                         'util_matrix': UtilMatrix, 'time_matrix': TimeMatrix, 'cost_matrix': CostMatrix,
                         'dwell_matrix': DwellArray, 'dist_matrix': DistMatrix}
            # start process
            process = mp.Process(target=SolverUtility.solver, args=(PenaltyQueue, idx, NodeNum, agent_database),
                                 kwargs=arg_input, name='P{}'.format(idx + 1))
            jobs.append(process)
            process.start()

    for j in jobs:
        # join process
        j.join()

    end_time = datetime.datetime.now()
    print('------ Evaluation time for initialization: {}s ------\n'.format((end_time - start_time).seconds))

    # 从 queue里提取parameter penalties
    Para_penalties_tuples = []
    while True:
        if PenaltyQueue.empty():  # 如果队列空了，就退出循环
            break
        else:
            Para_penalties_tuples.append(PenaltyQueue.get())

    Para_penalties = []
    for _i in range(len(s)):
        for _tuple in Para_penalties_tuples:
            if _i == _tuple[0]:
                Para_penalties.append(_tuple[1].penalty)  # Caution! 目前传回的tuple[1]是一个dict!!!
                break

    memo_parameter.extend(s)
    memo_penalty.extend(Para_penalties)
    print('\nmemo_parameter size" {}, memo_penalty size {}'.format(len(memo_parameter), len(memo_penalty)))

    PARAMETER[0] = s  # save parameters of each iteration into the PARAMETER dict.

    SCORES = list(penalty2score(Para_penalties)[0])  # functions returns ndarray

    # print evaluation scores
    print('Initialization: SCORES:')
    for i, _ in enumerate(SCORES):
        print(
            'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e' % (i + 1, s[i][0],
                                                                                        s[i][1],
                                                                                        s[i][2],
                                                                                        s[i][3],
                                                                                        _))

    for itr in range(itr_max):
        if itr > 0:
            print('------ Iteration {}\n'.format(itr + 1))
            # calculate scores and set record
            jobs = []
            PenaltyQueue = mp.Queue()  # queue, to save results for multi_processing

            # calculate evaluation time
            start_time = datetime.datetime.now()

            for idx, parameter in enumerate(s):
                print('Starting process {} in {}'.format(idx + 1, len(s)))

                # check existence of parameter in memory
                if parameter in memo_parameter:
                    PenaltyQueue.put((idx, memo_penalty[memo_parameter.index(parameter)]))

                    # 0722 加了这里。看看1. Penalty这个queue是不是按顺序添加paramter的；看看这个位置的值存在于paramater_memo吗？
                    print('Found the {} parameter in history.'.format(idx + 1, ))

                else:
                    ALPHA = parameter[:2]
                    BETA = [5] + parameter[2:]
                    arg_input = {'alpha': ALPHA, 'beta': BETA, 'phi': phi,
                                 'util_matrix': UtilMatrix, 'time_matrix': TimeMatrix, 'cost_matrix': CostMatrix,
                                 'dwell_matrix': DwellArray, 'dist_matrix': DistMatrix}
                    # start process

                    # 0722 Penalty这个Queue作为参数输入啦，所以肯定每个parameter的penalty都有对应地存入Penalty
                    # 0827 Results were not sent in order. Modified now.
                    process = mp.Process(target=SolverUtility.solver, args=(PenaltyQueue, idx, NodeNum, agent_database),
                                         kwargs=arg_input, name='P{}'.format(idx + 1))
                    jobs.append(process)
                    process.start()

            for j in jobs:
                # join process
                j.join()

            end_time = datetime.datetime.now()
            print(
                '------ Evaluation time for iteration{}: {} ------\n'.format(itr + 1, (end_time - start_time).seconds))

            # 从 queue里取值
            Para_penalties_tuples = []
            while True:
                if PenaltyQueue.empty():  # 如果队列空了，就退出循环
                    break
                else:
                    Para_penalties_tuples.append(PenaltyQueue.get())

            Para_penalties = []
            for _i in range(len(s)):
                for _tuple in Para_penalties_tuples:
                    if _i == _tuple[0]:
                        Para_penalties.append(_tuple[1])
                        break

            # 存入parameter memo
            memo_parameter.extend(s)
            memo_penalty.extend(Para_penalties)
            print('\nmemo_parameter size" {}, memo_penalty size {}'.format(len(memo_parameter), len(memo_penalty)))

            PARAMETER[itr] = s

            SCORES = list(penalty2score(Para_penalties)[0])  # functions returns ndarray

            # print evaluation scores
            print('Iteration {}: SCORES:'.format(itr + 1))
            for i, _ in enumerate(SCORES):
                print(
                    'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e' % (i + 1, s[i][0],
                                                                                                s[i][1],
                                                                                                s[i][2],
                                                                                                s[i][3],
                                                                                                _))

        # selection

        Indices = selection(inn, SCORES)
        print('\nIndices selected for iteration {}: {}\n'.format(itr + 1, np.array(Indices) + 1))

        s = list(s[_] for _ in Indices)
        SCORES = list(SCORES[_] for _ in Indices)  # s and SCORES should have same dimension

        print('Iteration {}: SCORES:\n'.format(itr + 1))
        for i, _ in enumerate(Indices):
            print(
                'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e' % (i + 1, s[i][0],
                                                                                            s[i][1],
                                                                                            s[i][2],
                                                                                            s[i][3],
                                                                                            SCORES[i]))

        # update parameter record
        if itr == 0:
            para_record = max(SCORES)  # duplicate 'record' use in the optimal solver module

        Best_score = max(SCORES)
        # debug print
        print('\nBest score: %.3e' % Best_score)
        print('Best score index: {}'.format(np.argsort(SCORES)[-1] + 1))
        print('x_max: {}'.format(np.array(s[np.argsort(SCORES)[-1]]).round(3)))
        print('\n检查下best score是否在score里面。')
        if Best_score > para_record:
            para_record = Best_score

        print('Record: %.3e; Best score in current generation: %.3e' % (para_record, Best_score))

        # write generation record and scores
        gnr_max.append(Best_score)
        y_mean.append(np.mean(SCORES))
        y_max.append(para_record)
        x_max.append(s[np.argsort(SCORES)[-1]])

        # mutation to produce next generation
        s = mutation(prob_mut, para_record, s, SCORES)  # mutation generates (inn + insertion size) individuals
        print('\nMutated parameters(individuals) for iteration {}: '.format(itr + 1))

        for i in range(len(s)):
            print(
                'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f' % (i + 1, s[i][0],
                                                                          s[i][1],
                                                                          s[i][2],
                                                                          s[i][3],
                                                                          ))
        print('\n')
        # %% plot
        if itr > 20:
            try:
                x = range(itr)
                fig = plt.figure(dpi=150)
                ax = fig.add_subplot(111)
                ax.plot(y_mean, color='lightblue')
                ax.plot(y_max, color='xkcd:orange')
                ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
                plt.ylabel("score")
                plt.xlabel('iteration')
                plt.xlim([0, itr + 1])
                plt.title("Parameter update process")
                plt.legend(['average', 'best'], loc='lower right', shadow=True, ncol=2)
                plt.show()
            except:
                pass

    # %% save results into DF
    Res = pd.DataFrame(
        columns=['itr', 'a1', 'a2', 'b2', 'b3', 'penalty', 'score', 'record_penalty', 'record', 'gnr_mean'])
    Res['itr'] = range(itr_max)
    Res.loc[:, 'a1':'b3'] = x_max
    Res['score'] = gnr_max
    Res['penalty'] = score2penalty(gnr_max)[0]
    Res['record_penalty'] = score2penalty(y_max)[0]
    Res['record'] = y_max
    Res['gnr_mean'] = y_mean
    Res.to_excel('Iteration result.xlsx')

    # save parameters into DF
    df_parameter = pd.DataFrame.from_dict(PARAMETER, orient='index')
    for i in range(df_parameter.shape[0]):
        for j in range(df_parameter.shape[1]):
            if df_parameter.loc[i, j]:
                df_parameter.loc[i, j] = [round(_, 3) for _ in np.array(df_parameter.loc[i, j])]

    df_parameter.to_excel('Parameter in each iteration.xlsx')
