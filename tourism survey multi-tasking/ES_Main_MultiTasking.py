"""This script is the main component in the research framework.
It is used to search for optimal behavioral parameter.
Dependence includes the 'slvr' package, data wrapping and so on.
Modified on Oct. 15. Last modified on Nov. 16"""

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


def evaluation(_s, _itr):
    """Evaluation of each population using MultiProcessing. Results are returned to the mp.queue in form of tuples."""
    global PARAMETER
    global memo_parameter, memo_penalty
    global phi, utility_matrix, dwell_vector, edge_time_matrix, edge_cost_matrix, edge_distance_matrix

    print('------ Iteration {} ------\n'.format(_itr + 1))

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # calculate evaluation time
    start_time = datetime.datetime.now()

    # evaluation with MultiProcessing for each parameter in current generation
    for idx, parameter in enumerate(_s):
        print('\nStarting process {} in {}'.format(idx + 1, len(_s)))
        parameter = parameter.tolist()  # convert to list
        # check existence of parameter in memory
        if parameter in memo_parameter:
            # sent back penalty tuple if exists in history
            penalty_queue.put((idx, memo_penalty[memo_parameter.index(parameter)]))
            print('\nThe {}th parameter is sent from history (with index {}), with score: {}'.format(
                idx, memo_parameter.index(parameter), memo_penalty[memo_parameter.index(parameter)]))
        else:
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
            process = mp.Process(target=SolverUtility.solver, args=(penalty_queue, idx, node_num, agent_database),
                                 kwargs=data_input, name='P{}'.format(idx + 1))
            jobs.append(process)
            process.start()

    for _j in jobs:
        # wait for processes to complete and join them
        _j.join()

    # collect end time
    end_time = datetime.datetime.now()
    print('\n------ Evaluation time for current iteration: {}s ------\n'.format((end_time - start_time).seconds))

    # retrieve parameter penalties from queue
    para_penalties_tuples = []
    while True:
        if penalty_queue.empty():  # 如果队列空了，就退出循环
            break
        else:
            para_penalties_tuples.append(penalty_queue.get())

    para_penalties = []
    # sort the retrieved penalties so that it has a same order with the original parameter set 's'
    for _i in range(len(_s)):
        for _tuple in para_penalties_tuples:
            if _i == _tuple[0]:
                para_penalties.append(_tuple[1])  # Caution! 目前传回的tuple[1]是一个dict!!!
                break

    memo_parameter.extend(_s.tolist())
    memo_penalty.extend(para_penalties)

    PARAMETER[_itr] = _s  # save parameters of each iteration into the PARAMETER dict.

    scores = penalty2score(para_penalties)[0]  # functions returns ndarray

    # print evaluation scores
    print('Evaluation scores for iteration {}:'.format(_itr))
    for _i, _ in enumerate(scores):
        print('Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e'
              % (_i + 1, _s[_i][0], _s[_i][1], _s[_i][2], _s[_i][3], _))
    return scores


def make_kid(pop, n_kid):
    # generate empty kid holder
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])

    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # crossover (roughly half p1 and half p2)
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)  # parent 1, parent 2

        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape) - 0.5), 0.)  # must > 0
        kv += ks * np.random.randn(*kv.shape)  # kv has size of 1 * DNA_SIZE

        # clip the mutated value so that the values fall in the boundaries
        kv[:] = list(np.clip(v, *DNA_BOUND[_i]) for _i, v in enumerate(kv))
        ''' kv[:]  = 和 kv = 的区别：  Originally kv refers to the elements in kids['DNA']. 
        后者把kv指代另一个变量了，而不改变zip里的element. 前者invokes slice assignment on the object kv refers to, 
        thus making the contents of the original object (element in kids['DNA']) a copy of the contents of new values'''

    """对于高维数组也是一样的
    x=np.array([[1,2,3,5,6,7,8,9],[1,2,3,5,6,7,8,9]])
    np.clip(x,3,8)

    Out[90]:
    array([[3, 3, 3, 5, 6, 7, 8, 8],
           [3, 3, 3, 5, 6, 7, 8, 8]])
    """
    return kids


def kill_bad(pop, _kids, _itr):
    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], _kids[key]))

    fitness = evaluation(pop['DNA'], _itr)  # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])  # create an array with range

    good_idx = idx[fitness.argsort()][-POP_SIZE:]  # selected by fitness ranking (not value)
    good_scores = fitness[fitness.argsort()][-POP_SIZE:]
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop, good_scores


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


# todo: mutation process is to be modified.
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
    # load agents
    with open(os.path.join(os.path.dirname(__file__), 'Database', 'transit_user_database.pickle'), 'rb') as file:
        agent_database = pickle.load(file)

    print('Setting up agents...')

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
    print('Starting travel_time_check...')
    for _ in range(3):
        while not no_update:
            print('Current iteration: {}'.format(itr + 1))
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
                print('Travel time update complete.\n')

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

    # %% behavioral parameters
    # setting up model coefficients. Agent preference data in agent_database.
    phi = 0.1

    # 'Evolution Strategy' parameter setup
    DNA_SIZE = 4  # DNA (real number)
    DNA_BOUND = [[-10, 0], [-10, 0], [0, 10], [0, 10]]  # solution upper and lower bounds
    N_GENERATIONS = 200
    POP_SIZE = 12  # population size (each individual in current generation is a vector of behavioral parameters)
    N_KID = 12  # n kids per generation

    memo_parameter, memo_penalty = [], []  # memo stores parameters from last 2 iterations
    PARAMETER = {}  # save parameters in each iteration

    y_mean, y_max, x_max, gnr_max = [], [], [], []  # 记录平均score, 每一世代max score， 每世代最佳个体
    # generate first population

    # generate first population. s = []  # species set of parameters
    # parameter = [-0.05, -0.05, 0.03, 0.1]
    # s = [[]]  # todo initialize s with good results in initialization evaluation

    s = [[1.0, 0.03, 0.3, 0.1],
         [1.0, 0.03, 1.0, 0.1],
         [0.03, 0.01, 3.0, 0.03],
         [1.0, 0.03, 3.0, 0.1],
         [1.0, 0.03, 0.1, 0.1],
         [1.0, 0.03, 0.03, 0.1],
         [1.0, 0.01, 0.1, 0.1],
         [1.0, 0.03, 0.01, 0.1],
         [3.0, 0.1, 1.0, 0.3],
         [1.0, 0.01, 0.01, 0.1],
         [0.3, 0.01, 0.3, 0.03]]

    for i in range(POP_SIZE - len(s)):  # to fill in s
        # random alphas
        a1, a2 = np.random.uniform(-0.5, -0.01), np.random.uniform(-0.5, -0.01)
        # random betas
        b2, b3 = np.random.uniform(0.01, 0.3), np.random.uniform(0.02, 1)
        s.append([a1, a2, b2, b3])

    mut_strength = [[0.2 * _ * np.random.rand() for _ in j] for j in s]  # 1/5 the DNA value

    POP = dict(DNA=np.array(s),  # initialize the pop DNA values
               mut_strength=np.array(mut_strength))  # initialize the pop mutation strength values

    # start iterations
    para_record = float('-inf')
    for iteration in range(N_GENERATIONS):
        # ES part
        kids = make_kid(POP, N_KID)
        POP, SCORES = kill_bad(POP, kids, iteration)  # keep some good parent for elitism
        # retrieve data for current iteration
        Best_score = max(SCORES)

        para_record = max(Best_score, para_record)  # duplicate 'record' use in the optimal solver module

        gnr_max.append(Best_score)
        y_mean.append(np.mean(SCORES))
        y_max.append(para_record)
        x_max.append(POP['DNA'][SCORES.argsort()][-1])  # pick the last one with highest score. x_max
        # print evaluation scores

        # %% plot
        if iteration > 20:
            try:
                x = range(iteration)
                fig = plt.figure(dpi=150)
                ax = fig.add_subplot(111)
                ax.plot(y_mean, color='lightblue')
                ax.plot(y_max, color='xkcd:orange')
                ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
                plt.ylabel("score")
                plt.xlabel('iteration')
                plt.xlim([0, iteration + 1])
                plt.title("Parameter update process")
                plt.legend(['average', 'best'], loc='lower right', shadow=True, ncol=2)
                plt.show()
            except:
                pass

    # %% save results into DF
    Res = pd.DataFrame(
        columns=['itr', 'a1', 'a2', 'b2', 'b3', 'penalty', 'score', 'record_penalty', 'record', 'gnr_mean'])
    Res['itr'] = range(N_GENERATIONS)
    Res.loc[:, 'a1':'b3'] = x_max
    Res['score'] = gnr_max
    Res['penalty'] = score2penalty(gnr_max)[0]
    Res['record_penalty'] = score2penalty(y_max)[0]
    Res['record'] = y_max
    Res['gnr_mean'] = y_mean
    Res.to_excel('{} iteration result.xlsx'.format(os.path.basename(__file__).split('.')[0]))

    # save parameters into DF
    for k, v in PARAMETER.items():
        PARAMETER[k] = v.tolist()
    df_parameter = pd.DataFrame.from_dict(PARAMETER, orient='index')
    for i in range(df_parameter.shape[0]):
        for j in range(df_parameter.shape[1]):
            if df_parameter.loc[i, j]:
                df_parameter.loc[i, j] = [round(_, 3) for _ in np.array(df_parameter.loc[i, j])]

    df_parameter.to_excel('Parameter in each iteration.xlsx')