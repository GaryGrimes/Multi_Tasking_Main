''' This script/module is to calculate the sums of penalty, i.e. "score" of a given behavioral parameter.
The score is a summation of estimation error between estiamted and observed paths, of each agent.
Each agent(tourist, in our case) in the data is enumerated and evaluated. Multi-Processing is utilitzed to speed up
computation as evaluating each agent is an indepentdent process.  '''

import multiprocessing as mp
import numpy as np
import os, pickle
import pandas as pd
import math
import datetime
from SolverUtility_ILS import SolverUtility


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


# split the arr into N chunks
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def eval_fun(s):  # enumerate all agents and calculate total errors
    # load agents
    with open(os.path.join(os.path.dirname(__file__), 'Database', 'transit_user_database.pickle'), 'rb') as file:
        agent_database = pickle.load(file)

    n_cores = mp.cpu_count()

    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # start process

    for idx, chunk in enumerate(pop):
        alpha = list(s[:2])
        beta = [5] + list(s[2:])
        data_input = {'alpha': alpha, 'beta': beta,
                      'phi': phi,
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}

        # enumerate all agents
        process = mp.Process(target=SolverUtility.solver, args=(penalty_queue, idx, node_num, chunk),
                             kwargs=data_input, name='P{}'.format(idx + 1))
        jobs.append(process)
        process.start()

    for j in jobs:
        # wait for processes to complete and join them
        j.join()

    # retrieve parameter penalties from queue
    penalty_total = 0
    while True:
        if penalty_queue.empty():  # 如果队列空了，就退出循环
            break
        else:
            penalty_total += penalty_queue.get()[1]  # 0是index，1才是data
    return penalty_total


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

# def evaluation(_s, _itr):
#     """Evaluation of each population using MultiProcessing. Results are returned to the mp.queue in form of tuples."""
#     global PARAMETER
#     global memo_parameter, memo_penalty
#     global phi, utility_matrix, dwell_vector, edge_time_matrix, edge_cost_matrix, edge_distance_matrix
#
#     print('------ Iteration {} ------\n'.format(_itr + 1))
#
#     jobs = []
#     penalty_queue = mp.Queue()  # queue, to save results for multi_processing
#
#     # calculate evaluation time
#     start_time = datetime.datetime.now()
#
#     # evaluation with MultiProcessing for each parameter in current generation
#     for idx, parameter in enumerate(_s):
#         print('\nStarting process {} in {}'.format(idx + 1, len(_s)))
#
#         # check existence of parameter in memory
#         if parameter in memo_parameter:
#             # sent back penalty tuple if exists in history
#             penalty_queue.put((idx, memo_penalty[memo_parameter.index(parameter)]))
#             print('\nThe {}th parameter is sent from history (with index {}), with score: {}'.format(
#                 idx, memo_parameter.index(parameter), memo_penalty[memo_parameter.index(parameter)]))
#         else:
#             ALPHA = list(parameter[:2])
#             BETA = [5] + list(parameter[2:])
#             data_input = {'alpha': ALPHA, 'beta': BETA,
#                           'phi': phi,
#                           'util_matrix': utility_matrix,
#                           'time_matrix': edge_time_matrix,
#                           'cost_matrix': edge_cost_matrix,
#                           'dwell_matrix': dwell_vector,
#                           'dist_matrix': edge_distance_matrix}
#
#             # start process
#             process = mp.Process(target=SolverUtility.solver, args=(penalty_queue, idx, node_num, agent_database),
#                                  kwargs=data_input, name='P{}'.format(idx + 1))
#             jobs.append(process)
#             process.start()
#
#     for j in jobs:
#         # wait for processes to complete and join them
#         j.join()
#
#     # collect end time
#     end_time = datetime.datetime.now()
#     print('\n------ Evaluation time for current iteration: {}s ------\n'.format((end_time - start_time).seconds))
#
#     # retrieve parameter penalties from queue
#     Para_penalties_tuples = []
#     while True:
#         if penalty_queue.empty():  # 如果队列空了，就退出循环
#             break
#         else:
#             Para_penalties_tuples.append(penalty_queue.get())
#
#     Para_penalties = []
#     # sort the retrieved penalties so that it has a same order with the original parameter set 's'
#     for _i in range(len(_s)):
#         for _tuple in Para_penalties_tuples:
#             if _i == _tuple[0]:
#                 Para_penalties.append(_tuple[1].penalty)
#                 break
#
#     memo_parameter.extend(_s)
#     memo_penalty.extend(Para_penalties)
#
#     PARAMETER[_itr] = _s  # save parameters of each iteration into the PARAMETER dict.
#
#     scores = penalty2score(Para_penalties)[0]  # functions returns ndarray
#
#     # print evaluation scores
#     print('Evaluation scores for iteration {}:'.format(_itr))
#     for _i, _ in enumerate(scores):
#         print('Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e'
#               % (_i + 1, _s[_i][0], _s[_i][1], _s[_i][2], _s[_i][3], _))
#     return scores


# idealy a module to feed parameter to completed processes. Since here we evaluate score for a single parameter, we
# simply divide the agent data to each process by the size of cpu core numbers.

# todo def
# arr是被分割的list，n是每个chunk中含n元素。


if __name__ == '__main__':
    pass
    # start_time = datetime.datetime.now()
    #
    # # test
    # num_cores = mp.cpu_count()
    #
    # s = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]
    # pop2 = chunks(s, num_cores)
    #
    # pop = chunks(agent_database, num_cores)
    # # for i in pop:
    # #     print(len(i))  # 尽可能平均
    #
    # jobs = []
    # penalty_queue = mp.Queue()  # queue, to save results for multi_processing
    #
    # # start process
    #
    # for idx, chunk in enumerate(pop):
    #     # # check existence of parameter in memory
    #     # if parameter in memo_parameter:
    #     #     # sent back penalty tuple if exists in history
    #     #     penalty_queue.put((idx, memo_penalty[memo_parameter.index(parameter)]))
    #     #     print('\nThe {}th parameter is sent from history (with index {}), with score: {}'.format(
    #     #         idx, memo_parameter.index(parameter), memo_penalty[memo_parameter.index(parameter)]))
    #     # else:
    #     ALPHA = list(s[:2])
    #     BETA = [5] + list(s[2:])
    #     data_input = {'alpha': ALPHA, 'beta': BETA,
    #                   'phi': phi,
    #                   'util_matrix': utility_matrix,
    #                   'time_matrix': edge_time_matrix,
    #                   'cost_matrix': edge_cost_matrix,
    #                   'dwell_matrix': dwell_vector,
    #                   'dist_matrix': edge_distance_matrix}
    #
    #     process = mp.Process(target=SolverUtility.solver, args=(penalty_queue, idx, node_num, chunk),
    #                          kwargs=data_input, name='P{}'.format(idx + 1))
    #     jobs.append(process)
    #     process.start()
    #
    # for j in jobs:
    #     # wait for processes to complete and join them
    #     j.join()
    #
    # # retrieve parameter penalties from queue
    # penalty_total = 0
    # while True:
    #     if penalty_queue.empty():  # 如果队列空了，就退出循环
    #         break
    #     else:
    #         penalty_total += penalty_queue.get()[1]  # 0是index，1才是data
    #
    # end_time = datetime.datetime.now()
    # print('\n------ Evaluation time for current iteration: {}s ------\n'.format((end_time - start_time).seconds))
    #
    # print('res: parameter {}, penalty: {}'.format(s, penalty_total))
