import numpy as np
import pickle
from ILS_master import IlsUtility
import os
import pandas as pd
import matplotlib.pyplot as plt

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


def path_similarity(path1, path2):
    pass


def penalty(particle):
    _answer = [-0.02, -0.01, 0.3, 0.1]
    diff = np.array(_answer) - np.array(particle)
    _penalty = np.exp(np.linalg.norm(diff))
    return _penalty


def selection(population, s_size):
    insertion_size = 5
    scores = []
    for _individual in population:
        scores.append(1 / penalty(_individual) ** 5)
    best_one_idx = np.argsort(scores)[-1]
    f_sum = sum(scores)
    prob = [_ / f_sum for _ in scores]
    # calculate accumulated prob
    prob_acu = [sum(prob[:_]) + prob[_] for _ in range(len(prob))]
    prob_acu[-1] = 1

    # return selected idx
    indices = []
    for _ in range(s_size - insertion_size):
        indices.append(next(x[0] for x in enumerate(prob_acu) if x[1] > np.random.rand()))
    indices.extend(insertion_size * [best_one_idx])
    return indices


def mutation(p, best_score, population, scores):
    insertion_size = 5
    learn_rate = [0.01, 0.01, 0.01, 0.02]
    species = []
    best = list(population[np.argsort(scores)[-1]])
    for _idx, _i in enumerate(population):
        if np.random.rand() < p:  # perform mutation, else pass
            _score = scores[_idx]
            weight = 4 * (np.abs(_score - best_score) / best_score)  # 0 <= weight < 5
            # alphas should < 0
            for _j, _par in enumerate(_i[:2]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par + _gain > 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])
                _par = _par + _gain  # update parameter
                _i[_j] = _par
            # betas should >= 0
            for _j, _par in enumerate(_i[2:]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _j])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par + _gain < 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _j])
                _par = _par + _gain  # update parameter
                _i[2 + _j] = _par
        species.append(_i)
    # insert the best solution so far
    """ always preserve the best solution """
    species.extend(insertion_size * [best])
    return species


# %% Solver Setup
NodeNum = 37  # number of attractions. Origin and destination are excluded.

Intrinsic_utilities = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Intrinsic Utility.xlsx'),
                                    sheet_name='data')
# node property
UtilMatrix = []
for _idx in range(Intrinsic_utilities.shape[0]):
    temp = np.around(list(Intrinsic_utilities.iloc[_idx, 1:4]), decimals=3)
    UtilMatrix.append(temp)

Dwell_time = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Dwell time array.xlsx'), index_col=0)
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
            print('Travel time update complete.')
Edge_cost_matrix = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'wide_transit_fare_matrix.csv'), index_col=0)
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

# parameter setup
phi = 0.1

inn = 10  # species size (each individual is our parameters here)
itr_max = 10
prob_mut = 0.8  # parameter update probability
""" memo is to store parameter --> Path similarity objective function"""

memo = []  # memo stores parameters from last 2 iterations
# TODO 新建 list 存储 last 3 iterations的parameter和value

s = []  # species set of parameters

# generate first population
""" beta 1 is fixed here"""
for i in range(inn):
    # random alphas
    a1, a2 = np.random.uniform(-0.5, -0.01), np.random.uniform(-0.5, -0.01)
    # random betas
    b2, b3 = np.random.uniform(0.01, 0.3), np.random.uniform(0.02, 1)
    s.append([a1, a2, b2, b3])
# manually insert reasonable parameters
s[-2], s[-1] = [-0.05, -0.05, 0.03, 0.1], [-0.03, -0.01, 0.02, 0.1]

y_mean, y_max, x_max, gnr_max = [], [], [], []  # 记录平均score, 每一世代max score， 每世代最佳个体

# calculate score and record of the 1st generation
print('------ Initialization first generation ------')
scores = []

for _id, parameter in enumerate(s):
    print('---- Evaluating parameter {} in {}'.format(_id, inn))
    ALPHA = parameter[:2]
    BETA = [5] + parameter[2:]  # TODO beta2该怎么定

    solver = IlsUtility(NodeNum, ALPHA, BETA, phi, UtilMatrix, TimeMatrix, CostMatrix, DwellArray)
    # solver.modify_travel_time()  # checked. reading from excel may cause some float number losing decimals?

    penalty = []

    for _idd, agent in enumerate(agent_database):
        if _idd > 0 and _idd % 100 == 0:
            print('--- Running optimal tours for the {} agent in {}'.format(_idd, len(agent_database)))
        Pref = agent.preference
        observed_path = agent.path_obs
        if Pref is None or observed_path is None:
            continue
        # skip empty paths (no visited location)
        if len(observed_path) < 3:
            continue

        # every path_op will be saved into the predicted path set for agent n
        Path_pdt = []
        # agent property
        Tmax, Origin, Destination = agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1
        # initialization
        init_res = solver.initialization(Tmax, Pref, Origin, Destination)
        if init_res is None or [] in init_res:
            continue
        else:
            PathOp, PathNop = init_res  # get value from a tuple

        Path_pdt.append(PathOp)

        # print('------  Scores after initial insertion: ------\n')
        # print('Optimal path score: %.2f, time: %d' % (solver.eval_util(PathOp, Pref), solver.time_callback(PathOp)))
        # print_path(PathOp)
        # for i in PathNop:
        #     print('Non-optimal path score: %.2f, time: %d' % (solver.eval_util(i, Pref), solver.time_callback(i)))
        #     print_path(i)

        # try different deviations
        for p in [0.05, 0.1, 0.15]:
            # print('\n------ Current deviation: {} ------'.format(p))
            record = solver.eval_util(PathOp, Pref)
            deviation = p * record
            best_solution = PathOp.copy()

            K = 3
            for _K in range(3):
                # print('\n------ Current K loop number: {} ------'.format(_K))
                for itr in range(4):
                    # print('\n---- Current iteration: {} ----'.format(itr))
                    # two-point exchange
                    Path_op, Path_nop = solver.two_point_exchange(PathOp, PathNop, Tmax, Pref, record, deviation)

                    visited = []
                    # print('\nScores after two-point exchange: \n')
                    score = solver.eval_util(Path_op, Pref)
                    # print('Optimal path score: %.2f, time: %d' % (score, solver.time_callback(Path_op)))
                    # print_path(Path_op)

                    if score > record:
                        Path_pdt.append(Path_op)
                        best_solution, record = list(Path_op), score
                        deviation = p * record

                    # one-point movement
                    Path_op, Path_nop = solver.one_point_movement(Path_op, Path_nop, Tmax, Pref, deviation, record)
                    visited = []

                    # print('\nScores after one-point movement: \n')
                    score = solver.eval_util(Path_op, Pref)
                    # print('Optimal path score: %.2f, time: %d' % (score, solver.time_callback(Path_op)))
                    # print_path(Path_op)
                    visited.extend(Path_op[1:-1])

                    if score > record:
                        Path_pdt.append(Path_op)
                        best_solution, record = list(Path_op), score
                        deviation = p * record

                    # 2-opt (clean-up)
                    # print('\nPath length before 2-opt: %d, with score: %.2f' % (solver.time_callback(Path_op),
                    #                                                             solver.eval_util(Path_op, Pref)))
                    Path_op_2 = solver.two_opt(Path_op, Pref)
                    cost_2_opt = solver.eval_util(Path_op_2, Pref)
                    # print('Path length after 2-opt: %d,  with score: %.2f' % (
                    # solver.time_callback(Path_op_2), cost_2_opt))

                    PathOp, PathNop = Path_op_2, Path_nop

                    # if no movement has been made, end I loop
                    if Path_op_2 == best_solution:
                        break
                    # if a new better solution has been obtained, then set new record and new deviation
                    if cost_2_opt > record:
                        Path_pdt.append(Path_op)
                        best_solution, record = list(Path_op_2), cost_2_opt
                        deviation = p * record
                # perform reinitialization
                PathOp, PathNop = solver.reinitialization(PathOp, PathNop, _K, Tmax, Pref)

            # print('\nBest solution score: %.2f, time: %d ' % (record, solver.time_callback(best_solution)))
            # print_path(best_solution)
            Path_pdt.append(best_solution)

        # compare predicted path with observed ones
        path_obs = list(np.array(agent.path_obs) - 1)
        best_path_predicted, lowest_penalty = [], 0
        for _path in Path_pdt:
            pnt = solver.path_penalty(path_obs, _path, DistMatrix)
            if not best_path_predicted:
                best_path_predicted, lowest_penalty = _path, pnt
            if pnt < lowest_penalty:
                best_path_predicted, lowest_penalty = _path, pnt
            # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref), solver.time_callback(_path)))
            # print('Penalty: {}'.format(res))
            # print_path(_path)

        # WRITE PREDICTED PATH AND PENALTY
        penalty.append(lowest_penalty)
        # TODO whether to save agents of each set of parameters...
        # agent.path_pdt = best_path_predicted
        # agent.penalty = lowest_penalty
        # path penalty evaluation

    # penalty is scaled... divided by 1,000
    scores.append(sum(penalty) / 1000)

# record = max(scores)
#
# for itr in range(itr_max):
#     # for idx, individual in enumerate(s):
#     #     print('{}: penalty: {}'.format(idx, penalty(individual)))
#
#     #  print (record)
#     Indices = selection(s, inn)
#
#     # selection
#     s = list(s[_] for _ in Indices)
#     if itr != 0:
#         # calculate scores and set record
#         scores = [1 / penalty(_) ** 5 for _ in s]
#
#     Best_score = max(scores)
#     if Best_score > record:
#         record = Best_score
#     gnr_max.append(Best_score)
#     y_mean.append(np.mean(scores))
#     y_max.append(record)
#     x_max.append(s[np.argsort(scores)[-1]])
#
#     # mutation
#     s = mutation(prob_mut, record, s, scores)
#
#     # save results
#
# # %% plot
#
# x = range(1, itr_max + 1)
# fig = plt.figure(dpi=150)
# ax = fig.add_subplot(111)
# ax.plot(x, y_mean, color='lightblue')
# ax.plot(y_max, color='xkcd:orange')
# plt.xlabel("# of iterations")
# # 显示纵轴标签
# plt.ylabel("score")
# plt.xlim([0, itr_max])
# plt.ylim([0, 1])
# # 显示图标题
# plt.title("Parameter update process")
# plt.legend(['average', 'best'], loc='best')
# plt.show()
