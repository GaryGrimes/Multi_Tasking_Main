import numpy as np
import numdifftools as nd
import os, pickle
import multiprocessing as mp
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy
from slvr.SolverUtility_ILS import SolverUtility


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


def f(beta):
    x, y = beta[0], beta[1]
    # return 2 * x ** 2 * y + 3 * y
    return 100 * (- x ** 2 - y ** 2 + 3)


def f_1(beta, pos):  # *loc 输入的是i,j i.e. 在哪两个方向上变化
    _center = list(beta)
    _epsilon = 0.01 * np.array(beta)  # todo 正式的code里epsilon是变化的，按array来定
    for _ in range(len(_epsilon)):
        _epsilon[_] = 0.0001 if _epsilon[_] < 0.0001 else _epsilon[_]
    _left, _right = list(_center), list(_center)
    _left[pos], _right[pos] = _left[pos] - _epsilon[pos], _right[pos] + _epsilon[pos]

    _l_res = (f(_left) - f(beta)) / -_epsilon[pos]
    _r_res = (f(_right) - f(beta)) / _epsilon[pos]
    return _l_res, _r_res


def f_2(beta, loc):
    d_1, d_2 = loc[0], loc[1]
    _epsilon = 0.01 * np.array(beta)
    for _ in range(len(_epsilon)):
        _epsilon[_] = 0.0001 if _epsilon[_] < 0.0001 else _epsilon[_]
    _left = list(beta)  # 仅取前差分算
    _left[d_2] -= _epsilon[d_2]
    return (f_1(_left, d_1)[0] - f_1(beta, d_1)[0]) / -_epsilon[d_2]  # 取前差分算


# 新建一个画布
figure = plt.figure()
# 新建一个3d绘图对象
ax = Axes3D(figure)

# 生成x, y 的坐标集 (-2,2) 区间，间隔为 0.1
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)

# 生成网格矩阵
X, Y = np.meshgrid(x, y)

# Z 轴 函数
Z = 1 * (- np.power(X, 2) - np.power(Y, 2) + 3)

# 定义x,y 轴名称
plt.xlabel("x")
plt.ylabel("y")

# 设置间隔和颜色
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
# 展示
plt.show()

if __name__ == '__main__':
    # %% Numerical Hessian for a simple analytical function
    x, y = sympy.symbols('x y')

    print('\nFunction f(x, y) = {}\n'.format(f([x, y])))

    # ------------- Calculation ------------- #
    B_star = [0.1, 0.01]

    # gradient
    Gradient = []
    for i in range(len(B_star)):
        Gradient.append(f_1(B_star, i))

    # calculate second derivative

    SecondDerivative = []
    # second gradient
    for i in range(len(B_star)):
        temp = []
        for j in range(len(B_star)):
            temp.append(f_2(B_star, [i, j]))
        SecondDerivative.append(temp)

    Gradient = np.array(Gradient).round(3)
    SecondDerivative = np.array(SecondDerivative).round(3)

    # ---------------------------- print results  ------------------------- #
    print('\nResult at point {}:\n'.format(B_star))
    print('The gradient of f(x,y) to x: backward: {}, forward: {}\n'.format(Gradient[0][0], Gradient[0][1]))
    print('The gradient of f(x,y) to y: backward: {}, forward: {}\n'.format(Gradient[1][0], Gradient[1][1]))

    # print second derivatives
    print('The second derivative matrix:\n {}'.format(SecondDerivative))

    variance = np.linalg.inv(-SecondDerivative)
    std_err = np.sqrt(np.diag(variance))

    print('The variance matrix:\n {}'.format(variance))
    print('The std errors:\n {}'.format(std_err))

    print('t value: {}'.format(np.array(B_star) / std_err))

    # print('点(1,2)处的二阶导dx2： ', f_2([1, 2], [0, 0]))
    #
    # print('点(3,5)处的二阶导： ', f_2([3, 5], [0, 0]))
    # print('点(0,0)处的二阶导： ', f_2([0, 0], [0, 0]))

    # ---------------------------- print results  ------------------------- #
    print('\nNumerical Gradients and Hessian using Numdifftools: \n')

    print('The gradient of f(x,y) to x and y: {}\n'.format(nd.Gradient(f)(B_star)))
    print('The Hessian (second derivative) matrix:\n {}'.format(nd.Hessian(f)(B_star)))

    num_hess = nd.Hessian(f)(B_star)

    variance = np.linalg.inv(-num_hess)
    std_err = np.sqrt(np.diag(variance))

    print('t value: {}'.format(np.array(B_star) / std_err))

    ''' 
    # numerical gradient using * parameter
    s = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]
    # # res_pa = nd.Gradient(pf.eval_fun)(s)
    # # res_test = pf.eval_fun(s)
    # # print('The score of parameter of {}: {}'.format(s, res_test))
    # res_test = pf.eval_fun([0, 0, 0, 0])

    # calculate average Hessian by emuerating through all agents and get their numerical Hessians

    # %%  Solver Database setup
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

    for _idx, _agent in enumerate(agent_database):
        try:
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
            penalty = SolverUtility.solver_single(node_num, _agent, **data_input)
            pass
        except ValueError:  # 万一func value为0，那就直接跳过该tourist
            print('Skipped at agent with index {}'.format(_idx))
            continue

    # todo 把node, edge properties都放在main里面
    
    '''
