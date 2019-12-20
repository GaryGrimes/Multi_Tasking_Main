"""Main module for parameter evaluation. Similar function with previous multi-tasking module
last modified: 11-15
"""

import numpy as np
import pickle
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from SimInfo import Solver_OP


# attributes (data)
class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


# methods


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # %% Solver setup
    #  Node properties
    node_num = 37

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

    node_properties = {'node_num': node_num,
                       'utility_matrix': utility_matrix,
                       'dwell_vector': dwell_vector}

    Solver_OP.node_setup(**node_properties)

    # %% Edge properties
    # Edge travel time

    # need several iterations to make sure direct travel is shorter than any detour
    Edge_time_matrix = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'transit_time_update2.xlsx'), index_col=0)
    edge_time_matrix = np.array(Edge_time_matrix)
    # see travel_time_check method in network module.

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

    edge_properties = {'edge_time_matrix': edge_time_matrix,
                       'edge_cost_matrix': edge_cost_matrix,
                       'edge_distance_matrix': edge_distance_matrix}

    Solver_OP.edge_setup(**edge_properties)

    # %% load agents
    with open(os.path.join(os.path.dirname(__file__), 'Database', 'transit_user_database.pickle'), 'rb') as file:
        agent_database = pickle.load(file)

    print('Setting up agents...')

    # %% setting up test parameters
    # todo check agent properties (attributes)
    Solver_OP.alpha = [-0.05, -0.05]
    Solver_OP.beta = [5, 0.03, 0.08]
    Solver_OP.phi = 0.1

    # %% setting up agent

    time_budget = 500  # person specific time constraints
    origin, destination = 28, 28
    visited = {}  # visited history
    preference = np.array([0.5, 0.3, 0.2])

    agent_properties = {'time_budget': time_budget,
                        'origin': origin,
                        'destination': destination,
                        'preference': preference,
                        'visited': {}}
    Solver_OP.agent_setup(**agent_properties)

    # %% start solver

    route = [29, 2, 4, 25]
    print('test %.2f \n' % Solver_OP.eval_util(route))

    # initialization
    PathOp, PathNop = Solver_OP.initialization()

    print('Scores after initialization (cost minimum insertion): \n')
    print('Optimal path score: {}, time: {}'.format(Solver_OP.eval_util(PathOp), Solver_OP.time_callback(PathOp)))
    print('Path_Op: {}'.format(PathOp))

    for i in PathNop:
        print('Non-optimal path score: {}, time: {}'.format(Solver_OP.eval_util(i), Solver_OP.time_callback(i)))
        print('Path: {}'.format(i))

    record, p = Solver_OP.eval_util(PathOp), 0.15
    deviation = p * record
    best_solution = PathOp.copy()
    K = 2

    for _K in range(K):
        print('\nCurrent K loop number: {}'.format(_K))
        for itr in range(4):
            print('\nCurrent iteration: {}'.format(itr))
            # two-point exchange
            Path_op, Path_nop = Solver_OP.two_point_exchange(PathOp, PathNop, record, deviation)
            visited = []
            print('\nScores after two-point exchange: \n')
            score = Solver_OP.eval_util(Path_op)
            print('Optimal path score: {}, time: {}'.format(score, Solver_OP.time_callback(Path_op)))
            print(Path_op)
            visited.extend(Path_op[1:-1])

            for i, path in enumerate(Path_nop):
                visited.extend(path[1:-1])
                print('Current path number: {}, score as {}, time: {}'.format(i, Solver_OP.eval_util(path),
                                                                              Solver_OP.time_callback(path)))
                print(path)

            print('Number of attractions visited: {}, duplicate nodes: {}.'.format(len(visited),
                                                                                   len(visited) - len(set(visited))))
            if score > record:
                best_solution, record = list(Path_op), score
                deviation = p * record

            # one-point movement
            Path_op, Path_nop = Solver_OP.one_point_movement(Path_op, Path_nop, deviation, record)
            visited = []

            print('\nScores after one-point movement: \n')
            score = Solver_OP.eval_util(Path_op)
            print('Optimal path score: {}, time: {}'.format(score, Solver_OP.time_callback(Path_op)))
            print(Path_op)
            visited.extend(Path_op[1:-1])

            if score > record:
                best_solution, record = list(Path_op), score
                deviation = p * record

            for i, path in enumerate(Path_nop):
                visited.extend(path[1:-1])
                print('Current path number: {}, score as {}, time: {}'.format(i, Solver_OP.eval_util(path),
                                                                              Solver_OP.time_callback(path)))
                print(path)

            print('Number of attractions visited: {}, duplicate nodes: {}.'.format(len(visited),
                                                                                   len(visited) - len(set(visited))))

            # 2-opt (clean-up)
            print('\nPath length before 2-opt: {}, with score: {}'.format(Solver_OP.time_callback(Path_op),
                                                                          Solver_OP.eval_util(Path_op)))
            Path_op_2 = Solver_OP.two_opt(Path_op)
            cost_2_opt = Solver_OP.eval_util(Path_op_2)
            print('Path length after 2-opt: {},  with score: {}'.format(Solver_OP.time_callback(Path_op_2), cost_2_opt))

            PathOp, PathNop = Path_op_2, Path_nop

            # if no movement has been made, end I loop
            if Path_op_2 == best_solution:
                break
            # if a new better solution has been obtained, then set new record and new deviation
            if cost_2_opt > record:
                best_solution, record = list(Path_op_2), cost_2_opt
                deviation = p * record
        # perform reinitialization
        PathOp, PathNop = Solver_OP.reinitialization(PathOp, PathNop, 3)

    print('\nBest solution score: {}, time: {} \nSolution: {}'.format(record, Solver_OP.time_callback(best_solution),
                                                                      best_solution))

    # test for beta sensitivity on Oct.30 2019

    # Order = [[0,1,2],[1,2,0],[2,1,0]]
    # for beta in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
    #     for order in Order:
    #         res = float('-inf')
    #         accu_util = np.zeros([1, 3])
    #         for k in order:
    #             util = A_util[k]
    #             exp_util = util * np.exp(-beta * accu_util)
    #             accu_util += exp_util
    #         print('Order:', order, 'beta:', beta, 'with utility: ', accu_util)
    #     print('\n')

    # modified on Nov. 3rd 2019
    print('\nPath penalty function test. Modified on Nov. 3rd 2019')
    test_pa, test_pb = [20, 19, 24, 23, 51, 20], [20, 19, 24, 20]  # node with indice > 47 included for test
    # test_pa, test_pb = [20, 20], [20, 19, 24, 20]
    # test_pa, test_pb = [20, 19, 24, 23, 20], [20, 20]
    # test_pa, test_pb = [20, 20], [20, 20]
    test_penalty = Solver_OP.distance_penalty(test_pa, test_pb)
    print('Modified evaluation function, test utility penalty is: {}'.format(test_penalty))
