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
        indices.append(next(_x[0] for _x in enumerate(prob_acu) if _x[1] > np.random.rand()))  # x is a tuple
    # insert best results from history
    indices.extend(insertion_size * [best_one_idx])
    return indices


def mutation(prob, best_score, population, population_scores):
    insertion_size = 5
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
            print('Travel time update complete.\n')
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

print('Setting up solver.')
# parameter setup
phi = 0.1

inn = 10  # species size (each individual is our parameters here)
itr_max = 100
prob_mut = 1  # parameter mutation probability
""" memo is to store parameter --> Path similarity objective function"""

memo_parameter, memo_penalty = [], []  # memo stores parameters from last 2 iterations

# s = []  # species set of parameters

# test s with fixed input
s = [[-0.37974139765941783,
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
      0.43319110214340956],
     [-0.05, -0.05, 0.03, 0.1],
     [-0.03, -0.01, 0.02, 0.1]]

# generate first population
""" beta 1 is fixed here"""
"""
for i in range(inn):
    # random alphas
    a1, a2 = np.random.uniform(-0.5, -0.01), np.random.uniform(-0.5, -0.01)
    # random betas
    b2, b3 = np.random.uniform(0.01, 0.3), np.random.uniform(0.02, 1)
    s.append([a1, a2, b2, b3])
# manually insert reasonable parameters
s[-2], s[-1] = [-0.05, -0.05, 0.03, 0.1], [-0.03, -0.01, 0.02, 0.1]
"""

y_mean, y_max, x_max, gnr_max = [], [], [], []  # 记录平均score, 每一世代max score， 每世代最佳个体

# calculate score and record of the 1st generation
print('------ Initialization: first generation ------')

PENALTIES = []
# evaluate scores for the first generation
for _id, parameter in enumerate(s):
    print('---- Evaluating parameter {} in {}'.format(_id + 1, inn))
    # check existence of parameter in memory
    if parameter in memo_parameter:
        PENALTIES.append(memo_penalty[memo_parameter.index(parameter)])
        continue

    ALPHA = parameter[:2]
    BETA = [5] + parameter[2:]

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
    PENALTIES.append(sum(penalty) / 1000)
    # save into memo
    memo_parameter.append(parameter)
    memo_penalty.append(sum(penalty) / 1000)

SCORES = list(penalty2score(PENALTIES)[0])  # functions returns ndarray
print('Initialization: SCORES: {}'.format(SCORES))
print('Initialization: species: {}'.format(s))
para_record = max(SCORES)  # duplicate 'record' use in the optimal solver module

for itr in range(itr_max):
    if itr > 0:
        print('------ Iteration {}\n'.format(itr))
        # calculate scores and set record
        PENALTIES = []
        # evaluate scores for the current generation s
        for _id, parameter in enumerate(s):
            print('---- Evaluating parameter {} in {}'.format(_id + 1, len(s)))
            # check existence of parameter in memory
            if parameter in memo_parameter:
                PENALTIES.append(memo_penalty[memo_parameter.index(parameter)])
                continue

            ALPHA = parameter[:2]
            BETA = [5] + parameter[2:]  #

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
                            Path_op, Path_nop = solver.two_point_exchange(PathOp, PathNop, Tmax, Pref, record,
                                                                          deviation)

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
                            Path_op, Path_nop = solver.one_point_movement(Path_op, Path_nop, Tmax, Pref, deviation,
                                                                          record)
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
            PENALTIES.append(sum(penalty) / 1000)
            # save into memo
            memo_parameter.append(parameter)
            memo_penalty.append(sum(penalty) / 1000)

        SCORES = list(penalty2score(PENALTIES)[0])  # functions returns ndarray

    Indices = selection(inn, SCORES)
    print('Indices selected for iteration {}: {}'.format(itr, Indices))

    print('Selected parameters(individuals): ')
    # selection
    s = list(s[_] for _ in Indices)
    print(s)
    SCORES = list(SCORES[_] for _ in Indices)  # s and SCORES should have same dimension
    print('With scores: ')
    print(SCORES)

    # update parameter record
    Best_score = max(SCORES)
    if Best_score > para_record:
        para_record = Best_score

    # write generation record and scores
    gnr_max.append(Best_score)
    y_mean.append(np.mean(SCORES))
    y_max.append(para_record)
    x_max.append(s[np.argsort(SCORES)[-1]])

    # mutation to produce next generation
    s = mutation(prob_mut, para_record, s, SCORES)  # mutation generates (inn + insertion size) individuals
    print('Mutated parameters(individuals) for iteration {}: '.format(itr))
    print(s)

# %% plot
x = range(itr_max)
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
ax.plot(x, y_mean, color='lightblue')
ax.plot(y_max, color='xkcd:orange')
plt.ylabel("score")
plt.xlim([0, itr_max])
plt.ylim([0, 1])
plt.title("Parameter update process")
plt.legend(['average', 'best'], loc='best')
plt.show()

# %% save results into DF
Res = pd.DataFrame(columns=['itr', 'a1', 'a2', 'b2', 'b3', 'penalty', 'score', 'record_penalty', 'record'])
Res['itr'] = range(itr_max)
Res.loc[:, 'a1':'b3'] = x_max
Res['score'] = gnr_max
Res['penalty'] = score2penalty(gnr_max)[0]
Res['record_penalty'] = score2penalty(y_max)[0]
Res['record'] = y_max
Res.to_excel('Iteration result.xlsx')
