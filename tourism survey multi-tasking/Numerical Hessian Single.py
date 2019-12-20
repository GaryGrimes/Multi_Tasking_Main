import numpy as np
import numdifftools as nd
import os, pickle
import multiprocessing as mp
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from slvr.SolverUtility_ILS import SolverUtility
import progressbar as pb


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


def obj_eval(s):  # enumerate all agents and calculate total errors
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
    return SolverUtility.solver_single(node_num, _agent, **data_input)


if __name__ == '__main__':
    #  Solver Database setup
    # B_star
    s = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]

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

    """ 工事中
    # todo  等 debug完成后再加入 multi-tasking
    # n_cores = mp.cpu_count()
    # 
    # pop = chunks(agent_database, n_cores)
    # # for i in pop:
    # #     print(len(i))  # 尽可能平均
    # 
    # jobs = []
    # penalty_queue = mp.Queue()  # queue, to save results for multi_processing
    # 
    # # start process
    # 
    # for idx, chunk in enumerate(pop):
    """
    p = pb.ProgressBar(widgets=[
        ' [', pb.Timer(), '] ',
        pb.Percentage(),
        ' (', pb.ETA(), ') ',
    ])

    p.start()
    total = len(agent_database)

    H = np.zeros([len(s), len(s)])

    for _idx, _agent in enumerate(agent_database):
        p.update(int((_idx / (total - 1)) * 100))
        try:
            if _idx > 8:
                break
            # res_pa = nd.Gradient(pf.eval_fun)(s)
            # res_test = pf.eval_fun(s)
            # print('The score of parameter of {}: {}'.format(s, res_test))
            # res_test = obj_eval(s)
            temp = nd.Hessian(obj_eval)(s)  # calculate Hessian for each single tourist
            H += temp  # add to Hessian for parameter
            # print('Current agent {} with prediction error {}'.format(_idx, res_test))
            pass
        except ValueError:  # 万一func value为0，那就直接跳过该tourist
            print('Skipped at agent with index {}'.format(_idx))
            continue

    Beta_Hessian = H/len(agent_database)  # in fact should be divided by successful users

    p.finish()

    # 最后H要除以n
    # todo 把node, edge properties都放在main里面
