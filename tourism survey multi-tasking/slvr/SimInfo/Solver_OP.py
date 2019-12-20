""" Module of the TTDP solver. Receive behavioral parameters and start up solver methods.
 Last modifed on Nov. 11 """

import numpy as np
import pandas as pd
import os
import Agent, Network

# behavioral parameters
alpha = []  # 1 * 2 vector
beta = []  # 1 * 3 vector
phi = 0  # float, time and monetary transfactor


# todo 增加方法，将修改network啊，agent的先通过solver修改

# -------- node setup -------- #

def node_setup(**kwargs):
    Network.node_num = kwargs['node_num']  # Number of attractions, excluding Origin and destination.
    Network.util_mat = kwargs['utility_matrix']
    Network.dwell_vec = kwargs['dwell_vector']


def edge_setup(**kwargs):
    Network.time_mat = kwargs['edge_time_matrix']  # Number of attractions, excluding Origin and destination.
    Network.cost_mat = kwargs['edge_cost_matrix']
    Network.dist_mat = kwargs['edge_distance_matrix']  # distance between attraction areas

    ''' Travel time check '''
    Network.travel_time_check()  # Prevent cases such that detours have shorter travel time than direct travel

    #  check UtilMatrix等各个matrix的shape是否正确。与NodeNum相符合
    if len(Network.util_mat) != Network.node_num:
        raise ValueError('Utility matrix error.')
    if Network.time_mat.shape[0] != Network.time_mat.shape[1]:
        raise ValueError('Time matrix error.')
    if Network.cost_mat.shape[0] != Network.cost_mat.shape[1]:
        raise ValueError('Cost matrix error.')
    if len(Network.dwell_vec) != Network.node_num:
        raise ValueError('Dwell time array error.')


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
    return _beta1 * np.dot(Agent.pref, Network.util_mat[to_node] * np.exp(-_beta2 * _accum_util)) + _beta3 * \
           Network.dwell_vec[to_node]


def exp_util_callback(to_node, _accum_util):
    global beta
    _beta_2 = beta[1]
    return Network.util_mat[to_node] * np.exp(-_beta_2 * _accum_util)


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


# TTDP methods

def initialization():
    o, d = Agent.Origin, Agent.Destination
    # for the points within ellipse, insert onto paths with cheapest insertion cost while ignoring the scores.
    distance = []
    for _node in range(Network.node_num):
        distance.append(Network.time_mat[o, _node] + Network.time_mat[_node, d] + Network.dwell_vec[_node])

    # check time limitation
    available_nodes = []
    for _, _dis in enumerate(distance):
        if _dis <= Agent.t_max and _ not in Agent.visited:
            available_nodes.append(_)

    L = min(10, len(available_nodes))
    # find L nodes with largest distance from start and end
    if L < 1:
        return None
    # index is node indices with distances from smallest to largest

    # build solutions. Reference: a fast and efficient heuristic for... Chao et al
    solutions = []
    path_op_set, path_nop_set = [], []
    for l in range(L):
        paths = []  # to store available paths (available nodes have to be on one of the paths)

        # construct 1st path
        cur_node_set = list(available_nodes)  # copy from [available_nodes]
        cur_path = [o, cur_node_set.pop(-(l + 1)), d]  # insert l-th largest node into the first path

        no_improvement = 0  # either path full (time limit exceeded) or no available nodes to be inserted
        while not no_improvement:
            cur_cost = time_callback(cur_path)  # regarding distance not score
            best_node, best_pos, best_cost = -1, -1, float('inf')
            for idx, node in enumerate(cur_node_set):
                for pos in range(1, len(cur_path)):  # check all positions on current path for insertion
                    _path = cur_path[:pos] + [node] + cur_path[pos:]
                    _cost = time_callback(_path) - cur_cost
                    if time_callback(_path) < Agent.t_max and _cost < best_cost:
                        best_node_idx, best_pos, best_cost = idx, pos, _cost
            no_improvement = best_cost == float('inf')
            if not no_improvement:
                cur_path = cur_path[:best_pos] + [cur_node_set.pop(best_node_idx)] + cur_path[best_pos:]
        paths.append(cur_path)

        # other paths
        # assign nodes to all paths
        while cur_node_set:
            cur_path = [o, cur_node_set.pop(0),
                        d]  # cur_node_set is already sorted, the first node is with smallest distance from o to d
            no_improvement = 0
            while not no_improvement:
                cur_cost = time_callback(cur_path)  # regarding distance not score
                best_node, best_pos, best_cost = -1, -1, float('inf')
                for idx, node in enumerate(cur_node_set):
                    for pos in range(1, len(cur_path)):  # check all positions on current path for insertion
                        _path = cur_path[:pos] + [node] + cur_path[pos:]
                        _cost = time_callback(_path) - cur_cost
                        if time_callback(_path) < Agent.t_max and _cost < best_cost:
                            best_node_idx, best_pos, best_cost = idx, pos, _cost
                no_improvement = best_cost == float('inf')
                if not no_improvement:
                    cur_path = cur_path[:best_pos] + [cur_node_set.pop(best_node_idx)] + cur_path[best_pos:]
            paths.append(cur_path)

        # decide the solution path by choosing a path with largest total score among the paths
        _score, solution = [eval_util(_path) for _path in paths], []
        if _score:
            solution = paths.pop(np.argsort(_score)[-1])
        path_op_set.append(solution)
        path_nop_set.append(paths)

    # return best path_op and its path_nop set
    best_op, best_nop = [], []
    if path_op_set:
        _score = [eval_util(_path) for _path in path_op_set]
        best_op, best_nop = path_op_set[np.argsort(_score)[-1]], path_nop_set[np.argsort(_score)[-1]]
    return best_op, best_nop


def two_point_exchange(path_op, path_nop, _record, _deviation):
    _TimeMatrix = Network.time_mat
    _DwellArray = Network.dwell_vec

    cur_op, cur_nop = list(path_op), list(path_nop)
    a_loop_nodes = list(cur_op[1:-1])
    # A loop
    for idx, node_j in enumerate(a_loop_nodes):  # first to the last point in path_op (excluding o and d)
        # for debugger in cur_nop:
        #     debugger_time = self.time_callback(debugger)
        #     if debugger_time > tmax:
        #         raise LookupError('Path time over limit.')
        cur_op_backup = cur_op.copy()  # the point remain in current position if exchange results in a worse score

        _ = cur_op[1:-1]
        _.remove(node_j)
        cur_op = [cur_op[0]] + _ + [
            cur_op[-1]]  # o + attractions + d. Sometimes origin or des will also exist in attractions

        # node_j is removed from cur_op
        length = time_callback(cur_op)

        found = 0  # Flag to indicate whether a candidate exchange leading to a higher total score is found.
        # If found, the exchange is performed immediately, and all other exchanges are ignored.

        # B loop
        # b_loop_records = []
        # b_loop_scores = []
        exchange_flag = 0
        best_path_idx, best_path, best_node, best_pos, best_score, = float('-inf'), [], -1, -1, float('-inf')
        for _path_idx, _path in enumerate(cur_nop):
            if found == 1:
                break
            for index in range(1, len(_path) - 1):
                node_i = _path[index]
                # skip node_j and duplicate node
                if node_i == node_j or node_i in cur_op:  # avoid duplicate
                    continue
                for pos in range(1, len(cur_op)):
                    # feasibility check
                    if _TimeMatrix[cur_op[pos - 1]][node_i] + _DwellArray[node_i] + _TimeMatrix[node_i][cur_op[pos]] - \
                            _TimeMatrix[cur_op[pos - 1]][cur_op[pos]] + length < Agent.t_max:
                        test_path = cur_op[:pos] + [node_i] + cur_op[pos:]
                        test_score = eval_util(test_path)
                        # find best insert position
                        if test_score >= best_score:
                            best_path_idx, best_path, best_node, best_pos, best_score = _path_idx, _path, node_i, pos, test_score

                # do the exchange
                if best_path:  # found an insertion location indeed
                    # total score increase check
                    if best_score > _record:
                        found = 1  # found an exchange that leads to a higher score
                        # exchange
                        cur_op = cur_op[:best_pos] + [best_node] + cur_op[best_pos:]
                        best_path.pop(index)
                        exchange_flag = 1
                        break
            # b_loop_records.append([best_node, best_pos])
            # b_loop_scores.append(best_score)
            # b_loop ends

        # if found no exchange, try exchanges between record and (record - deviation)
        if found == 0:
            if best_path:
                test_path = cur_op[:best_pos] + [best_node] + cur_op[best_pos:]
                test_score = eval_util(test_path)
            else:
                test_path, test_score = [], float('-inf')

            if test_score >= _record - _deviation:
                # exchange
                # insert node_i onto cur_op
                cur_op = cur_op[:best_pos] + [best_node] + cur_op[best_pos:]
                # remove node_i from the best_path in path_nop
                visits = list(best_path[1:-1])
                visits.remove(best_node)
                cur_nop[best_path_idx] = [best_path[0]] + visits + [best_path[-1]]
                exchange_flag = 1
            pass

        # if found no exchange, cur_op remains the same
        if not exchange_flag:
            cur_op = cur_op_backup
            # no removing nodes from path_nop
            continue

        # put node_j back into cur_nop
        # criteria: minimum insertion cost
        best_path_idx, best_path, best_pos, best_score = float('inf'), [], -1, float('inf')
        for bp_idx, _path in enumerate(cur_nop):
            if node_j in _path[1:-1]:  # skip nodes that serve as origin or destination
                raise LookupError('Duplicate nodes are not supposed to present! Debug please.')
                # continue  # avoid repetitive existence
            for pos in range(1, len(_path)):
                length = time_callback(_path)
                # feasibility check
                if _TimeMatrix[_path[pos - 1]][node_j] + _DwellArray[node_j] + _TimeMatrix[node_j][_path[pos]] - \
                        _TimeMatrix[_path[pos - 1]][_path[pos]] + length < Agent.t_max:
                    test_path = _path[:pos] + [node_j] + _path[pos:]
                    test_score = time_callback(test_path) - length
                    # find best insert position
                    if test_score <= best_score:
                        best_path_idx, best_path, best_pos, best_score = bp_idx, _path, pos, test_score
                        # do the exchange
        if not best_score == float('inf'):  # found an insertion location indeed
            # TODO check if change is made inplace
            cur_nop[best_path_idx] = best_path[:best_pos] + [node_j] + best_path[best_pos:]
        else:
            # construct new path into cur_nop
            new_path = [path_op[0], node_j, path_op[-1]]
            cur_nop.append(new_path)

    # pick up best from both path_op and path_nop
    solutions = [cur_op] + cur_nop
    # DEBUG
    best_score, best_path, best_index = float('-inf'), [], float('-inf')
    for index, solution in enumerate(solutions):
        if len(set(solution[1:-1])) < len(solution[1:-1]):
            raise LookupError('Duplicate nodes in a path')
        cur_score = eval_util(solution)
        if cur_score > best_score:
            best_path, best_score, best_index = solution, cur_score, index
    p_op = solutions.pop(best_index)
    p_nop = solutions
    return p_op, p_nop


def one_point_movement(path_op, path_nop, _deviation, _record):
    # calculate points that are within ellipse
    o, d = path_op[0], path_op[-1]

    distance = []
    for _node in range(Network.node_num):
        distance.append(Network.time_mat[o][_node] + Network.time_mat[_node][d] + Network.dwell_vec[_node])
    # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
    index = np.argsort(distance)
    # check time limitation and visit history
    available_nodes = [x for x in index if distance[x] <= Agent.t_max and x not in Agent.visited]

    paths = [path_op] + path_nop  # paths变了源变量也跟着变

    for _node in available_nodes:
        # pick out the current path that the node is on:
        path_q = []
        for _i, _path in enumerate(paths):
            if _node in _path[1:-1]:
                path_q = paths.pop(_i)
                break
        # movement
        movement = 0
        best_path_index, best_pos, best_score = float('-inf'), -1, float('-inf')
        for path_index, _path in enumerate(paths):
            for pos in range(1, len(_path)):
                test_path = _path[:pos] + [_node] + _path[pos:]
                # check feasibility:
                if time_callback(test_path) < Agent.t_max:
                    test_score = eval_util(test_path)
                    # check total score increase
                    # if test_score > eval_util(_path):  # TODO total score here 是指每个path的还是record的？
                    if test_score > _record:
                        paths[path_index] = test_path  # do movement
                        _ = path_q[1:-1]
                        _.remove(_node)
                        if len(_) == len(path_q[1:-1]):
                            raise LookupError('Remove not successful... not found node in current path?')
                        path_q, movement = [path_q[0]] + _ + [path_q[-1]], 1
                        # TODO 这里可能是inplace的, 能不能这样放回到paths里？  list()后应该就可以
                        paths = [list(path_q)] + paths
                        break
                        #
                    else:
                        if test_score > best_score:
                            best_path_index, best_pos, best_score = path_index, pos, test_score
            if movement:
                break
        if movement == 0:
            # check if the score of the best movement >= record - deviation
            if best_score >= _record - _deviation:
                # make movement
                paths[best_path_index] = paths[best_path_index][:best_pos] + [_node] + paths[best_path_index][
                                                                                       best_pos:]
                # delete current node on path_q
                _ = path_q[1:-1]
                _.remove(_node)
                if len(_) == len(path_q[1:-1]):
                    raise LookupError('Remove not successful... not found node in current path?')
                path_q, movement = [path_q[0]] + _ + [path_q[-1]], 1
                paths = [list(path_q)] + paths
            else:
                paths = [list(path_q)] + paths  # put path_q back if no movement
    score = []
    for _path in paths:
        score.append(eval_util(_path))
    path_op = paths.pop(np.argsort(score)[-1])
    return path_op, paths


def two_opt(path_op):
    best = list(path_op)
    _record = eval_util(best)

    temp = []
    improved = True
    while improved:
        improved = False
        for _i in range(1, len(path_op) - 2):
            for j in range(_i + 1, len(path_op)):
                if j - _i == 1:
                    continue
                if util_change(best[_i - 1], best[_i], best[j - 1], best[j]) > 0.001:  # if travel utility increases
                    best[_i:j] = best[j - 1:_i - 1:-1]
                    temp.append(best)
                    improved = True
    # choose a path with highest utility
    res, _score = path_op, _record
    for _ in temp:  # check improvement of utility
        if eval_util(_) > _score and time_callback(_) <= Agent.t_max:
            res = _
    return res


def reinitialization(path_op, path_nop, _k):
    if _k < 1 or _k > len(path_op) - 2:
        return path_op, path_nop
    ratio = []
    # visited = path_op[1:-1]
    for _idx in range(1, len(path_op) - 1):
        # si/costi
        gain = node_util_callback(path_op[_idx], np.zeros([3]))
        cost = Network.time_mat[path_op[_idx - 1], path_op[_idx]] + Network.dwell_vec[path_op[_idx]] + \
               Network.time_mat[path_op[_idx],
                                path_op[_idx + 1]] - Network.time_mat[path_op[_idx - 1], path_op[_idx + 1]]
        ratio.append(gain / cost)
    # ratio is benefit/insertion cost
    nodes_sorted = np.argsort(ratio)  # smaller nodes are sorted at front
    remove_indices = nodes_sorted[:_k]

    # for _i, _node in enumerate(path_op):
    path_op_new = [path_op[x] for x in range(len(path_op)) if x - 1 not in remove_indices]
    for _k in range(_k):
        remove_node_idx = remove_indices[_k] + 1
        # remove node from path_op
        node_j = path_op[remove_node_idx]  # TODO 每次path_op都在变小，不能按idx pop
        # put node_j back into path_nop
        # criteria: minimum insertion cost
        best_path_idx, best_path, best_pos, best_score = float('inf'), [], -1, float('inf')
        for bp_idx, _path in enumerate(path_nop):
            if node_j in _path[1:-1]:  # skip nodes that serve as origin or destination
                raise LookupError('Duplicate nodes are not supposed to present! Debug please.')
                # continue  # avoid repetitive existence
            for pos in range(1, len(_path)):
                length = time_callback(_path)
                # feasibility check
                if Network.time_mat[_path[pos - 1], node_j] + Network.dwell_vec[node_j] + \
                        Network.time_mat[node_j, _path[pos]] - Network.time_mat[_path[pos - 1], _path[pos]] + \
                        length < Agent.t_max:
                    test_path = _path[:pos] + [node_j] + _path[pos:]
                    test_score = time_callback(test_path) - length
                    # find best insert position
                    if test_score <= best_score:
                        best_path_idx, best_path, best_pos, best_score = bp_idx, _path, pos, test_score
                        # do the exchange
        if not best_score == float('inf'):  # found an insertion location indeed
            path_nop[best_path_idx] = best_path[:best_pos] + [node_j] + best_path[best_pos:]
        else:
            # construct new path into cur_nop
            new_path = [path_op[0], node_j, path_op[-1]]
            path_nop.append(new_path)
    path_op = path_op_new
    return path_op, path_nop


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


def distance_penalty(p_a, p_b):  # created on Nov.3 2019
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
    pass
