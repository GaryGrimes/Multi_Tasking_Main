"""Main module for parameter evaluation. Similar function with previous multi-tasking module
last modified: 11-15
"""

import numpy as np
import pickle
import os
import pandas as pd
import datetime
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from SimInfo import Solver_ILS
from SimInfo import Solver_OP
import progressbar as pb


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
    # Solver setup

    # %% plot config.
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # %% setting up test parameters
    # todo check agent properties (attributes)
    Solver_ILS.alpha = Solver_OP.alpha = [-0.05, -0.05]
    Solver_ILS.beta = Solver_OP.beta = [5, 0.03, 0.08]
    Solver_ILS.phi = Solver_OP.phi = 0.1

    # %% setting up agents. First set up agents, then nodes, then edges
    with open(os.path.join(os.path.dirname(__file__), 'Database', 'transit_user_database.pickle'), 'rb') as file:
        agent_database = pickle.load(file)

    print('Setting up agents...')
    time_budget = 200  # person specific time constraints
    origin, destination = 40, 28
    visited = {}  # visited history
    preference = np.array([0.5, 0.3, 0.2])

    agent_properties = {'time_budget': time_budget,
                        'origin': origin,
                        'destination': destination,
                        'preference': preference,
                        'visited': {}}

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

    Solver_ILS.edge_setup(**edge_properties)
    Solver_OP.edge_setup(**edge_properties)

    # %% start solver
    Solver_ILS.agent_setup(**agent_properties)
    Solver_ILS.node_setup(**node_properties)

    route = [29, 2, 4, 25]
    print('Utility for the test path: %.2f \n' % Solver_ILS.eval_util(route))

    InitialPath = Solver_ILS.initial_solution()

    if len(InitialPath) <= 2:
        FinalOrder = InitialPath
    else:
        firstVisit = InitialPath[1]
        Solver_ILS.Node_list[firstVisit].visit = 1

        Order = InitialPath
        FinalOrder = list(Order)

        # No edgeMethod in my case
        U, U8, U10 = [], [], []
        S, R = 1, 1
        v = 2  # TODO v means # of visits (excluding origin)?

        counter2 = 0
        NoImprove = 0
        BestFound = float('-inf')

        while NoImprove < 50:
            BestScore = float('-inf')
            LocalOptimum = 0

            # print(Order)

            while LocalOptimum == 0:
                LocalOptimum, Order, BestScore = Solver_ILS.insert(Order, BestScore)

            counter2 += 1  # 2指inner loop的counter
            v = len(Order) - 1

            U.append(BestScore)  # TODO U is utility memo
            U8.append(v)
            U10.append(max(U))

            if BestScore > BestFound:
                BestFound = BestScore
                FinalOrder = list(Order)

                NoImprove = 0  # improved
            else:
                NoImprove += 1

            if len(Order) <= 2:
                continue
            else:
                s = np.random.randint(1, len(Order) - 1)
                R = np.random.randint(1, len(Order) - s)  # from node S, delete R nodes (including S)

            if s >= min(U8):
                s = s - min(U8) + 1

            Order = Solver_ILS.shake(Order, s, R)

            # S += R
            # R += 1
    print('Near optimal path: {}, with total time {} min, utility {}.'.format(FinalOrder,
                                                                              Solver_ILS.time_callback(FinalOrder),
                                                                              Solver_ILS.eval_util(FinalOrder)))

    # %% solve benchmark test
    p = pb.ProgressBar(widgets=[
        ' [', pb.Timer(), '] ',
        pb.Percentage(),
        ' (', pb.ETA(), ') ',
    ])

    # size, scores and execution time
    time_budget_bounds = [200, 800]
    iteration_size = 200
    ILS_scores, OTS_scores = [], []
    ILS_time, OTS_time = [], []
    p.start()

    for _time in range(*time_budget_bounds, 10):

        # ------ ILS solver ------ #

        # time for ILS solver
        start_time = datetime.datetime.now()

        time_budget = _time  # person specific time constraints
        origin, destination = 40, 28
        visited = {}  # visited history
        preference = np.array([0.5, 0.3, 0.2])

        agent_properties = {'time_budget': time_budget,
                            'origin': origin,
                            'destination': destination,
                            'preference': preference,
                            'visited': {}}

        # setup agents and nodes
        Solver_ILS.agent_setup(**agent_properties)
        Solver_ILS.node_setup(**node_properties)

        InitialPath = Solver_ILS.initial_solution()
        if len(InitialPath) <= 2:
            FinalOrder = InitialPath
        else:
            firstVisit = InitialPath[1]
            Solver_ILS.Node_list[firstVisit].visit = 1

            Order = InitialPath
            FinalOrder = list(Order)

            # No edgeMethod in my case
            U, U8, U10 = [], [], []
            S, R = 1, 1
            v = 2  # TODO v means # of visits (excluding origin)?

            counter2 = 0
            NoImprove = 0
            BestFound = float('-inf')

            while NoImprove < 50:
                BestScore = float('-inf')
                LocalOptimum = 0

                # print(Order)

                while LocalOptimum == 0:
                    LocalOptimum, Order, BestScore = Solver_ILS.insert(Order, BestScore)

                counter2 += 1  # 2指inner loop的counter
                v = len(Order) - 1

                U.append(BestScore)  # TODO U is utility memo
                U8.append(v)
                U10.append(max(U))

                if BestScore > BestFound:
                    BestFound = BestScore
                    FinalOrder = list(Order)

                    NoImprove = 0  # improved
                else:
                    NoImprove += 1

                if len(Order) <= 2:
                    continue
                else:
                    s = np.random.randint(1, len(Order) - 1)
                    R = np.random.randint(1, len(Order) - s)  # from node S, delete R nodes (including S)

                if s >= min(U8):
                    s = s - min(U8) + 1

                Order = Solver_ILS.shake(Order, s, R)

                # S += R
                # R += 1

        end_time = datetime.datetime.now()

        cur_ILS_score = Solver_ILS.eval_util(FinalOrder)
        ILS_scores.append(cur_ILS_score)

        ILS_time.append((end_time - start_time))

        # OTS solver
        start_time = datetime.datetime.now()

        agent_properties = {'time_budget': time_budget,
                            'origin': origin,
                            'destination': destination,
                            'preference': preference,
                            'visited': {}}
        Solver_OP.agent_setup(**agent_properties)

        # initialization
        PathOp, PathNop = Solver_OP.initialization()

        record, prob = Solver_OP.eval_util(PathOp), 0.15
        deviation = prob * record
        best_solution = PathOp.copy()
        K = 2

        for _K in range(K):
            for itr in range(4):
                # two-point exchange
                Path_op, Path_nop = Solver_OP.two_point_exchange(PathOp, PathNop, record, deviation)
                visited = []
                score = Solver_OP.eval_util(Path_op)
                visited.extend(Path_op[1:-1])

                for i, path in enumerate(Path_nop):
                    visited.extend(path[1:-1])

                if score > record:
                    best_solution, record = list(Path_op), score
                    deviation = prob * record

                # one-point movement
                Path_op, Path_nop = Solver_OP.one_point_movement(Path_op, Path_nop, deviation, record)
                visited = []

                score = Solver_OP.eval_util(Path_op)
                visited.extend(Path_op[1:-1])

                if score > record:
                    best_solution, record = list(Path_op), score
                    deviation = prob * record

                for i, path in enumerate(Path_nop):
                    visited.extend(path[1:-1])

                # 2-opt (clean-up)

                Path_op_2 = Solver_OP.two_opt(Path_op)
                cost_2_opt = Solver_OP.eval_util(Path_op_2)

                PathOp, PathNop = Path_op_2, Path_nop

                # if no movement has been made, end I loop
                if Path_op_2 == best_solution:
                    break
                # if a new better solution has been obtained, then set new record and new deviation
                if cost_2_opt > record:
                    best_solution, record = list(Path_op_2), cost_2_opt
                    deviation = prob * record
            # perform reinitialization
            PathOp, PathNop = Solver_OP.reinitialization(PathOp, PathNop, 3)

        end_time = datetime.datetime.now()
        OTS_scores.append(record)

        OTS_time.append((end_time - start_time))

        print('Time budget: {}, ILS score: {} vs. OTS score {}'.format(_time, cur_ILS_score, record))

        p.update(int((_time - time_budget_bounds[0]) / (time_budget_bounds[1] - 200 - 1)) * 100)

    p.finish()

    # %% optimal score evaluation
    iteration_size, Optimal_score = 200, []

    for _time in range(*time_budget_bounds, 10):
        print('Solving optimal score for time budget: {}.\n'.format(_time))
        # ILS solver
        cur_score = []

        p.start()

        for _itr in range(iteration_size):
            if input('Continue to calculate the optimal score? (execution time will be long.) '
                     'press: y/n').strip() != 'y':
                break

            time_budget = _time  # person specific time constraints
            origin, destination = 40, 28
            visited = {}  # visited history
            preference = np.array([0.5, 0.3, 0.2])

            agent_properties = {'time_budget': time_budget,
                                'origin': origin,
                                'destination': destination,
                                'preference': preference,
                                'visited': {}}

            # setup agents and nodes
            Solver_ILS.agent_setup(**agent_properties)
            Solver_ILS.node_setup(**node_properties)

            InitialPath = Solver_ILS.initial_solution()
            if len(InitialPath) <= 2:
                FinalOrder = InitialPath
            else:
                firstVisit = InitialPath[1]
                Solver_ILS.Node_list[firstVisit].visit = 1

                Order = InitialPath
                FinalOrder = list(Order)

                # No edgeMethod in my case
                U, U8, U10 = [], [], []
                S, R = 1, 1
                v = 2  # TODO v means # of visits (excluding origin)?

                counter2 = 0
                NoImprove = 0
                BestFound = float('-inf')

                while NoImprove < 50:
                    BestScore = float('-inf')
                    LocalOptimum = 0

                    # print(Order)

                    while LocalOptimum == 0:
                        LocalOptimum, Order, BestScore = Solver_ILS.insert(Order, BestScore)

                    counter2 += 1  # 2指inner loop的counter
                    v = len(Order) - 1

                    U.append(BestScore)  # TODO U is utility memo
                    U8.append(v)
                    U10.append(max(U))

                    if BestScore > BestFound:
                        BestFound = BestScore
                        FinalOrder = list(Order)

                        NoImprove = 0  # improved
                    else:
                        NoImprove += 1

                    if len(Order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(Order) - 1)
                        R = np.random.randint(1, len(Order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(U8):
                        s = s - min(U8) + 1

                    Order = Solver_ILS.shake(Order, s, R)

                    # S += R
                    # R += 1
            cur_score.append(Solver_ILS.eval_util(FinalOrder))

            p.update(int((_itr) / (iteration_size - 1)) * 100)
        Optimal_score.append(max(cur_score))
        p.finish()

    # %% plot benchmark test
    # execution time performance
    x = np.arange(*time_budget_bounds, 10)
    ILS_time_val, OTS_time_val = list(map(lambda x: x.total_seconds(), ILS_time)), \
                                 list(map(lambda x: x.total_seconds(), OTS_time))
    plt.figure(dpi=200)
    plt.plot(x, ILS_time_val, 'b--', label='Iterated Local Search')
    plt.plot(x, OTS_time_val, 'k--', label='OP-based')

    plt.xlabel('Time Budget')
    plt.ylabel('Execution Time (sec.)')
    plt.legend(loc='best')
    plt.show()

    # %% score performance

    plt.figure(dpi=200)
    plt.plot(x, Optimal_score, 'r-.', label='Iterated Local Search (200 iter.)')
    plt.plot(x, ILS_scores, 'b--', label='Iterated Local Search (average)')
    plt.plot(x, OTS_scores, 'k--', label='OP-based')

    plt.xlabel('Time budget')
    plt.ylabel('Optimal path score')
    plt.legend(loc='best')
    plt.show()

    # %% draw histogram of the benchmark test
    #
    # """
    #     Demo of the histogram (hist) function with a few features.
    #
    #     In addition to the basic histogram, this demo shows a few optional features:
    #
    #         * Setting the number of data bins
    #         * The ``normed`` flag, which normalizes bin heights so that the integral of
    #           the histogram is 1. The resulting histogram is a probability density.
    #         * Setting the face color of the bars
    #         * Setting the opacity (alpha value).
    #
    #     """
    # # example data
    # mu = 100  # mean of distribution
    # sigma = 15  # standard deviation of distribution
    # x = mu + sigma * np.random.randn(10000)
    #
    # num_bins = 50
    # # the histogram of the data
    # n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
    #
    # # add a 'best fit' line
    # y = mlab.normpdf(bins, mu, sigma)
    #
    # plt.plot(bins, y, 'r--')
    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
    #
    # # Tweak spacing to prevent clipping of ylabel
    # plt.subplots_adjust(left=0.15)
    # plt.show()
    #
    # # %%
    # # normed=True是频率图，默认是频数图
    # plt.hist(x, bins=30, normed=False,
    #          weights=None, cumulative=False, bottom=None,
    #          histtype=u'bar', align=u'left', orientation=u'vertical',
    #          rwidth=0.9, log=False, color=None, label=None, stacked=False,
    #          )
    #
    # plt.xlabel('Optimal path score', fontsize=12)
    # plt.ylabel('Percentage (%)', fontsize=12)
    # plt.title('Histogram of score distribution (total: 200)', fontsize=12)
    #
    # # plt.legend(fontsize=12)
    # fig = plt.gcf()
    # fig.set_size_inches(7.2, 4.2)
    # plt.show()
