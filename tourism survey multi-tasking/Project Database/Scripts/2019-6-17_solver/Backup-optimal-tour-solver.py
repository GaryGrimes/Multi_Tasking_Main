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
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'transit_time_wide_update.xlsx'), index_col=0)
Edge_cost_matrix = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'wide_transit_fare_matrix.csv'), index_col=0)

TimeMatrix, CostMatrix = np.array(Edge_time_matrix), np.array(Edge_cost_matrix)  # time in min

# TODO check UtilMatrix等各个matrix的shape是否正确。与nodenum相符合
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

# %% start solver
# TODO warning: total utility of a path must >= 0

# parameter setup
phi = 0.1

ALPHA = [-0.05, -0.05]
BETA = [5, 0.03, 0.1]  # TODO beta2该怎么定

solver = IlsUtility(NodeNum, ALPHA, BETA, phi, UtilMatrix, TimeMatrix, CostMatrix, DwellArray)
solver.modify_travel_time()

# %% start solver
# warning: total utility of a path must >= 0

Agents = [agent_database[1]]

for agent in Agents:
    Pref = agent.preference
    observed_path = agent.path_obs
    if Pref is None or observed_path is None:
        continue
    # every path_op will be saved into the predicted path set for agent n
    Path_pdt = []
    # agent property
    Tmax, Origin, Destination = agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1
    # initialization
    PathOp, PathNop = solver.initialization(Tmax, Pref, Origin, Destination)

    Path_pdt.append(PathOp)

    print('------  Scores after initial insertion: ------\n')
    print('Optimal path score: %.2f, time: %d' % (solver.eval_util(PathOp, Pref), solver.time_callback(PathOp)))
    print_path(PathOp)
    # for i in PathNop:
    #     print('Non-optimal path score: %.2f, time: %d' % (solver.eval_util(i, Pref), solver.time_callback(i)))
    #     print_path(i)

    # try different deviations
    for p in [0.05, 0.1, 0.15]:
        print('\n------ Current deviation: {} ------'.format(p))
        record = solver.eval_util(PathOp, Pref)
        deviation = p * record
        best_solution = PathOp.copy()

        K = 3
        for _K in range(3):
            print('\n------ Current K loop number: {} ------'.format(_K))
            for itr in range(4):
                print('\n---- Current iteration: {} ----'.format(itr))
                # two-point exchange
                Path_op, Path_nop = solver.two_point_exchange(PathOp, PathNop, Tmax, Pref, record, deviation)

                visited = []
                print('\nScores after two-point exchange: \n')
                score = solver.eval_util(Path_op, Pref)
                print('Optimal path score: %.2f, time: %d' % (score, solver.time_callback(Path_op)))
                print_path(Path_op)

                if score > record:
                    Path_pdt.append(Path_op)
                    best_solution, record = list(Path_op), score
                    deviation = p * record

                # one-point movement
                Path_op, Path_nop = solver.one_point_movement(Path_op, Path_nop, Tmax, Pref, deviation, record)
                visited = []

                print('\nScores after one-point movement: \n')
                score = solver.eval_util(Path_op, Pref)
                print('Optimal path score: %.2f, time: %d' % (score, solver.time_callback(Path_op)))
                print_path(Path_op)
                visited.extend(Path_op[1:-1])

                if score > record:
                    Path_pdt.append(Path_op)
                    best_solution, record = list(Path_op), score
                    deviation = p * record

                # 2-opt (clean-up)
                print('\nPath length before 2-opt: %d, with score: %.2f' % (solver.time_callback(Path_op),
                                                                            solver.eval_util(Path_op, Pref)))
                Path_op_2 = solver.two_opt(Path_op, Pref)
                cost_2_opt = solver.eval_util(Path_op_2, Pref)
                print('Path length after 2-opt: %d,  with score: %.2f' % (solver.time_callback(Path_op_2), cost_2_opt))

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
            PathOp, PathNop = solver.reinitialization(PathOp, PathNop, 3, Tmax, Pref)

        print('\nBest solution score: %.2f, time: %d ' % (record, solver.time_callback(best_solution)))
        print_path(best_solution)
        Path_pdt.append(best_solution)
