""" Module of the TTDP solver. Receive behavioral parameters and start up solver methods.
 Last modifed on Nov. 11 """

import numpy as np
import pandas as pd
import os
import Agent, Network

# behavioral parameters
alpha = []  # 1 * 2 vector
beta = []  # 1 * 3 vector
phi = 0  # float, to transform monetary unit into time

# instances of nodes and egdes contains solver methods. Thus they are in the same category with solver methods.
Node_list = []
Edge_list = []


# -------- node setup -------- #

def node_setup(**kwargs):
    global Node_list

    Network.node_num = kwargs['node_num']  # Number of attractions, excluding Origin and destination.
    Network.util_mat = kwargs['utility_matrix']
    Network.dwell_vec = kwargs['dwell_vector']

    # for each agent do a node setup. Thus the global variable Node_list and Edge list should be initialized for each agent.
    Node_list = []
    for count in range(Network.node_num):
        x = Network.Node(Network.dwell_vec[count], Network.util_mat[count], count)
        x.visit = 1 if count in Agent.visited else 0
        Node_list.append(x)


def edge_setup(**kwargs):
    global Edge_list
    Network.time_mat = kwargs['edge_time_matrix']  # Number of attractions, excluding Origin and destination.
    Network.cost_mat = kwargs['edge_cost_matrix']
    Network.dist_mat = kwargs['edge_distance_matrix']  # distance between attraction areas

    Edge_list = []
    for origin in range(Network.time_mat.shape[0]):
        edge_list = []
        for destination in range(Network.time_mat.shape[1]):
            x = Network.Edge(phi, origin, destination, Network.time_mat[origin, destination],
                             Network.cost_mat[origin, destination], Network.dist_mat[origin, destination])
            edge_list.append(x)
        Edge_list.append(edge_list)


def agent_setup(**kwargs):
    Agent.t_max = kwargs['time_budget']  # person specific time constraints
    Agent.Origin = kwargs['origin']
    Agent.Destination = kwargs['destination']
    Agent.pref = kwargs['preference']
    Agent.visited = kwargs['visited']


def arc_util_callback(from_node, to_node):
    global alpha, phi

    _alpha_1, _alpha_2, _phi = alpha[0], alpha[1], phi

    return _alpha_1 * Network.time_mat[from_node, to_node] + _alpha_2 * _phi * Network.cost_mat[from_node, to_node]


def node_util_callback(to_node, _accum_util):
    global beta

    [_beta1, _beta2, _beta3] = beta
    # always presume a negative discount factor
    return _beta1 * np.dot(Agent.pref, Network.util_mat[to_node] * np.exp(min(-_beta2, 0) * _accum_util)) + _beta3 * \
           Network.dwell_vec[to_node]


def exp_util_callback(to_node, _accum_util):
    global beta
    _beta_2 = beta[1]
    # always presume a negative discount factor
    return Network.util_mat[to_node] * np.exp(min(-_beta_2, 0) * _accum_util)


def eval_util(_route):  # use array as input
    _pref = Agent.pref
    res, _accum_util = 0, np.zeros([3])
    if len(_route) <= 2:
        return float("-inf")
    else:
        _k = 1
        for _k in range(1, len(_route) - 1):
            # arc and node utility
            res += arc_util_callback(_route[_k - 1], _route[_k]) + node_util_callback(_route[_k], _accum_util)
            _accum_util += exp_util_callback(_route[_k], _accum_util)  # Accumulated utility; travel history
        res += arc_util_callback(_route[_k], _route[_k + 1])
        return res
    pass


def travel_time_callback(from_node, to_node):
    return Network.time_mat[from_node, to_node]


def time_callback(_route):
    _DwellArray = Network.dwell_vec
    if len(_route) <= 2:
        return 0
    else:
        _time, _k = 0, 1
        for _k in range(1, len(_route) - 1):
            _time += travel_time_callback(_route[_k - 1], _route[_k]) + _DwellArray[_route[_k]]
        _time += travel_time_callback(_route[_k], _route[_k + 1])
        return _time


def cost_change(self, n1, n2, n3, n4):
    cost_matrix = self.costmatrix
    return cost_matrix[n1][n3] + cost_matrix[n2][n4] - cost_matrix[n1][n2] - cost_matrix[n3][n4]


def util_change(n1, n2, n3, n4):  # utility improvement if result is positive
    return arc_util_callback(n1, n3) + arc_util_callback(n2, n4) - arc_util_callback(n1, n2) - arc_util_callback(n3, n4)


def initial_solution():
    o, d, t_max = Agent.Origin, Agent.Destination, Agent.t_max
    distance, benefit = [], []
    for _i in range(Network.node_num):
        cost = Network.time_mat[o, _i] + Network.time_mat[_i, d] + Network.dwell_vec[_i]
        distance.append(cost)
        _benefit = np.dot(Agent.pref, Network.util_mat[_i]) / cost
        benefit.append(_benefit)
    # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
    index = list(np.argsort(benefit))
    # except for node_j

    # check time limitation
    available_nodes = [_x for _x in index if distance[_x] <= t_max][::-1]  # nodes with higher benefits at front
    if not available_nodes:
        return []
    # randomly pick an available node for insertion
    _ = [o, available_nodes[np.random.randint(len(available_nodes))], d]
    return _


def comp_fill():
    o, d, t_max = Agent.Origin, Agent.Destination, Agent.t_max
    distance, benefit = [], []
    for _i in range(Network.node_num):
        cost = Network.time_mat[o, _i] + Network.time_mat[_i, d] + Network.dwell_vec[_i]
        distance.append(cost)
        _benefit = np.dot(Agent.pref, Network.util_mat[_i]) / cost
        benefit.append(_benefit)
    # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
    index = list(np.argsort(benefit))

    # compulsory fill-in
    return [o, index[-1], d]



def insert(order, best_score):
    t_max = Agent.t_max
    local_optimum = 0
    best_node, best_pos = None, None
    check = best_score  # score record

    _feasibility = 1
    for ii in range(len(Node_list)):
        cur_node = Node_list[ii]
        if cur_node.visit == 0:
            for jj in range(1, len(order)):
                path_temp = order[:jj] + [ii] + order[jj:]
                # check time budget feasibility
                _feasibility = time_callback(path_temp) < t_max
                # calculate utility and save best score and best position
                if _feasibility:
                    _utility = eval_util(path_temp)
                    if _utility > best_score:
                        best_score, best_node, best_pos = _utility, ii, jj
                pass

    if best_score > check:
        order = order[:best_pos] + [best_node] + order[best_pos:]
        for ii in range(1, len(order) - 1):
            Node_list[order[ii]].visit = 1
    else:
        local_optimum = 1
    return local_optimum, order, best_score


def shake(order, s, r):
    path_temp = order[:s] + order[s + r:]
    # delete node visits
    for _ in range(s, s + r):
        Node_list[order[_]].visit = 0
    return path_temp


def path_penalty(p_a, p_b):
    distance_matrix = Network.dist_mat
    # path_a and path_b both starts from 0. offset starts from 0, i.e., 0 --> 1st destination
    if p_a[0] != p_b[0] or p_a[-1] != p_b[-1]:
        raise ValueError('Paths have different o or d.')

    # define insertion cost
    o, d = p_a[0], p_a[-1]
    insertion_cost = [distance_matrix[o][_] + distance_matrix[_][d] for _ in range(distance_matrix.shape[0])]

    # check empty path
    path_a, path_b = p_a[1:-1], p_b[1:-1]
    if not path_a or not path_b:
        if not path_a and not path_b:  # if all empty
            return 0
        elif path_a:  # path b is empty
            try:
                _ = sum([max(distance_matrix[x]) for x in path_a])  # 19-10-03: take the largest distance
            except IndexError:
                _ = 0
            return _
        else:  # path a is empty
            try:
                _ = sum([max(distance_matrix[x]) for x in path_b])  # calculate most distant results
            except IndexError:
                _ = 0
            return _

    # if both paths are not empty (excluding o, d)

    # check node indices. Observed path with Detailed location (58) or unclear places (99) were skipped.
    # TODO Better to omit those places in the path than skipping to next tourist
    max_idx = max(max(path_a), max(path_b))
    if max_idx > min(distance_matrix.shape) - 1:
        return 0

    rows, cols = len(path_a) + 1, len(path_b) + 1

    # the editing distance matrix
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    # source prefixes can be transformed into empty strings

    # by deletions:
    for row in range(1, rows):
        dist[row][0] = dist[row - 1][0] + insertion_cost[path_a[row - 1]]
    # target prefixes can be created from an empty source string

    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = dist[0][col - 1] + insertion_cost[path_b[col - 1]]

    for col in range(1, cols):
        for row in range(1, rows):
            deletes = insertion_cost[path_a[row - 1]]
            inserts = insertion_cost[path_b[col - 1]]
            subs = distance_matrix[path_a[row - 1]][path_b[col - 1]]  # dist from a to b

            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + subs)  # substitution
    return dist[row][col]

    # TODO case when path length equal


def geo_dist_penalty(p_a, p_b):  # created on Nov.3 2019
    """Pa is observed path, Pb predicted"""

    # Offset is 0 for the 1st destination.
    distance_matrix = Network.dist_mat
    if p_a[0] != p_b[0] or p_a[-1] != p_b[-1]:
        raise ValueError('Paths have different o or d.')

    # define the penalty in utility form for every two destinations. u_ik stands for the generalized cost of travel
    o, d = p_a[0], p_a[-1]

    path_a, path_b = p_a[1:-1], p_b[1:-1]  # excluding origin and destination

    path_node_check = []
    for _path in [path_a, path_b]:
        _new_path = []
        for node in _path:
            if node <= min(distance_matrix.shape) - 1:
                _new_path.append(node)
        path_node_check.append(_new_path)
    path_a, path_b = path_node_check[0], path_node_check[1]

    # utility (negative) penalty evaluation
    cost, a, b = 0, o, o  # let a, b be origin

    # if exist empty path
    if not path_a:  # if observed path is empty
        return cost

    while path_a and path_b:
        a, b = path_a.pop(0), path_b.pop(0)  # a, b correspond to the i_th node in path_a, path_b
        cost += distance_matrix[a][b]

    if path_a:  # length of path_a > path b
        while path_a:
            a = path_a.pop(0)
            cost += distance_matrix[a][b]
    else:  # case when length of path_b > path a
        while path_b:
            b = path_b.pop(0)
            cost += distance_matrix[a][b]
    return cost


if __name__ == '__main__':
    # %% create node instances
    # assign values to node instances

    pass
