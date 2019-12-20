import numpy as np
import pickle


class IlsUtility(object):
    def __init__(self, node_num, alpha, beta, global_phi, util_mat, time_mat, cost_mat, dwell_arr):
        if len(alpha) != 2:
            raise ValueError('alpha should be a 1*2 array!')
        if len(beta) != 3:
            raise ValueError('beta should be a 1*3 array!')
        self.NodeNum = node_num
        self.alpha, self.beta, self.phi = alpha, beta, global_phi
        self.utilmatrix = util_mat
        self.timematrix = time_mat
        self.costmatrix = cost_mat
        self.dwellarray = dwell_arr

    def modify_travel_time(self):
        timematrix = self.timematrix
        flag = 0
        for i in range(timematrix.shape[0]):
            for j in range(timematrix.shape[1]):
                min_cost = timematrix[i][j]
                for k in range((timematrix.shape[1])):
                    cost = timematrix[i][k] + timematrix[k][j]
                    if cost < min_cost:
                        print('travel time error　from {} to {}.'.format(i, j))
                    # min_cost = cost if cost < min_cost else min_cost
                timematrix[i][j] = min(min_cost, timematrix[i][j])
        self.timematrix = (timematrix + timematrix.T) / 2

    def eval_util(self, route, pref):  # use array as input
        util, AcumUtil = 0, np.zeros([3])
        if len(route) <= 2:
            return -999999
        else:
            for k in range(1, len(route) - 1):
                # arc and node utility
                util += self.arc_util_callback(route[k - 1], route[k]) + self.node_util_callback(route[k], pref,
                                                                                                 AcumUtil)
                AcumUtil += self.exp_util_callback(route[k], AcumUtil)  # Accumulated utility; travel history
            util += self.arc_util_callback(route[k], route[k + 1])
            return util
        pass

    def cost_change(self, n1, n2, n3, n4):
        cost_matrix = self.costmatrix
        return cost_matrix[n1][n3] + cost_matrix[n2][n4] - cost_matrix[n1][n2] - cost_matrix[n3][n4]

    def time_callback(self, route):
        DwellArray = self.dwellarray
        if len(route) <= 2:
            return 0
        else:
            time = 0
            for k in range(1, len(route) - 1):
                time += self.travel_time_callback(route[k - 1], route[k]) + DwellArray[route[k]]
            time += self.travel_time_callback(route[k], route[k + 1])
            return time

    def travel_time_callback(self, from_node, to_node):
        return self.timematrix[from_node][to_node]

    def arc_util_callback(self, from_node, to_node):
        alpha1, alpha2, phi = self.alpha[0], self.alpha[1], self.phi
        return alpha1 * self.timematrix[from_node][to_node] + alpha2 * phi * self.costmatrix[from_node][to_node]

    def exp_util_callback(self, to_node, AcumUtil):
        beta2 = self.beta[1]
        return self.utilmatrix[to_node] * np.exp(-beta2 * AcumUtil)

    def node_util_callback(self, to_node, pref, AcumUtil):
        [beta1, beta2, beta3] = self.beta
        return beta1 * np.dot(pref, self.utilmatrix[to_node] * np.exp(-beta2 * AcumUtil)) + beta3 * self.dwellarray[
            to_node]

    def util_max_insert(self, o, d, tmax, pref, must_node=None):
        if must_node:
            _path = [o, must_node, d]
        else:
            _path = [o, d]
        # construct new path into cur_nop
        distance, benefit = [], []
        for _i in range(self.NodeNum):
            cost = self.timematrix[o][_i] + self.timematrix[_i][d] + self.dwellarray[_i]
            distance.append(cost)
            bene = np.dot(pref, self.utilmatrix[_i]) / cost
            benefit.append(bene)
        # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
        index = list(np.argsort(benefit))
        # except for node_j
        if must_node in index:
            index.remove(must_node)
        # check time limitation
        available_nodes = [x for x in index if distance[x] <= tmax][::-1]  # nodes with higher benefits at front
        while available_nodes:
            # try all available nodes, even if current node cannot be inserted due to big cost
            cur_node = available_nodes.pop(0)
            min_cost = 999999
            for k in range(1, len(_path)):  # available positions for insertion
                # try all possible insertions
                newpath = _path[:k] + [cur_node] + _path[k:]
                newcost = self.time_callback(newpath)
                if newcost < tmax and newcost < min_cost:
                    min_cost, bespos = newcost, k
            nochange = min_cost == 999999
            if not nochange:
                _path = _path[:bespos] + [cur_node] + _path[bespos:]
        return _path

    def initialization(self, tmax, pref, o, d):
        # for the points within ellipse, insert onto paths with cheapest insertion cost while ignoring the scores.
        distance = []
        for _node in range(self.NodeNum):
            distance.append(self.timematrix[o][_node] + self.timematrix[_node][d] + self.dwellarray[_node])
        # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
        index = np.argsort(distance)
        # check time limitation
        available_nodes = [x for x in index if distance[x] <= tmax]
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
            cur_node_set = list(available_nodes)
            cur_path = [o, cur_node_set.pop(-(l + 1)), d]  # insert l-th largest node into the first path

            no_improvement = 0  # either path full (time limit exceeded) or no available nodes to be inserted
            while not no_improvement:
                cur_cost = self.time_callback(cur_path)  # regarding distance not score
                best_node, best_pos, best_cost = -1, -1, 999999
                for idx, node in enumerate(cur_node_set):
                    for pos in range(1, len(cur_path)):  # check all positions on current path for insertion
                        _path = cur_path[:pos] + [node] + cur_path[pos:]
                        _cost = self.time_callback(_path) - cur_cost
                        if self.time_callback(_path) < tmax and _cost < best_cost:
                            best_node_idx, best_pos, best_cost = idx, pos, _cost
                no_improvement = best_cost == 999999
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
                    cur_cost = self.time_callback(cur_path)  # regarding distance not score
                    best_node, best_pos, best_cost = -1, -1, 999999
                    for idx, node in enumerate(cur_node_set):
                        for pos in range(1, len(cur_path)):  # check all positions on current path for insertion
                            _path = cur_path[:pos] + [node] + cur_path[pos:]
                            _cost = self.time_callback(_path) - cur_cost
                            if self.time_callback(_path) < tmax and _cost < best_cost:
                                best_node_idx, best_pos, best_cost = idx, pos, _cost
                    no_improvement = best_cost == 999999
                    if not no_improvement:
                        cur_path = cur_path[:best_pos] + [cur_node_set.pop(best_node_idx)] + cur_path[best_pos:]
                paths.append(cur_path)

            # decide the solution path by choosing a path with largest total score among the paths
            score, solution = [self.eval_util(_path, pref) for _path in paths], []
            if score:
                solution = paths.pop(np.argsort(score)[-1])
            path_op_set.append(solution)
            path_nop_set.append(paths)

        # return best path_op and its path_nop set
        best_op, best_nop = [], []
        if path_op_set:
            score = [self.eval_util(_path, pref) for _path in path_op_set]
            best_op, best_nop = path_op_set[np.argsort(score)[-1]], path_nop_set[np.argsort(score)[-1]]
        return best_op, best_nop

    def two_point_exchange(self, path_op, path_nop, tmax, pref, record, deviation):
        TimeMatrix = self.timematrix
        DwellArray = self.dwellarray
        cur_op, cur_nop = list(path_op), list(path_nop)
        a_loop_nodes = list(cur_op[1:-1])
        # A loop
        for idx, node_j in enumerate(a_loop_nodes):  # first to the last point in path_op (except for o and d)
            # for debugger in cur_nop:
            #     debugger_time = self.time_callback(debugger)
            #     if debugger_time > tmax:
            #         raise LookupError('Path time over limit.')
            cur_op_backup = cur_op.copy()  # the point remain in current position if exchange results in a bad score

            _ = cur_op[1:-1]
            _.remove(node_j)
            cur_op = [cur_op[0]] + _ + [
                cur_op[-1]]  # o + attractions + d. Sometimes origin or des will also exist in attractions

            # node_j is removed from cur_op
            length = self.time_callback(cur_op)

            found = 0  # Flag to indicate whether a candidate exchange leading to a higher total score is found.
            # If found, the exchange is performed immediately, and all other exchanges are ignored.

            # B loop  TODO 加入best path，能在没有找到最优解的情况按deviation修改
            # b_loop_records = []
            # b_loop_scores = []
            exchange_flag = 0
            best_path_idx, best_path, best_node, best_pos, best_score, = -999999, [], -1, -1, -999999
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
                        if TimeMatrix[cur_op[pos - 1]][node_i] + DwellArray[node_i] + TimeMatrix[node_i][cur_op[pos]] - \
                                TimeMatrix[cur_op[pos - 1]][cur_op[pos]] + length < tmax:
                            test_path = cur_op[:pos] + [node_i] + cur_op[pos:]
                            test_score = self.eval_util(test_path, pref)
                            # find best insert position
                            if test_score >= best_score:
                                best_path_idx, best_path, best_node, best_pos, best_score = _path_idx, _path, node_i, pos, test_score

                    # do the exchange
                    if best_path:  # found an insertion location indeed
                        # total score increase check
                        if best_score > record:
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
                    test_score = self.eval_util(test_path, pref)
                else:
                    test_path, test_score = [], -999999

                if test_score >= record - deviation:
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
            best_path_idx, best_path, best_pos, best_score = 999999, [], -1, 999999
            for bp_idx, _path in enumerate(cur_nop):
                if node_j in _path[1:-1]:  # skip nodes that serve as origin or destination
                    raise LookupError('Duplicate nodes are not supposed to present! Debug please.')
                    # continue  # avoid repetitive existence
                for pos in range(1, len(_path)):
                    length = self.time_callback(_path)
                    # feasibility check
                    if TimeMatrix[_path[pos - 1]][node_j] + DwellArray[node_j] + TimeMatrix[node_j][_path[pos]] - \
                            TimeMatrix[_path[pos - 1]][_path[pos]] + length < tmax:
                        test_path = _path[:pos] + [node_j] + _path[pos:]
                        test_score = self.time_callback(test_path) - length
                        # find best insert position
                        if test_score <= best_score:
                            best_path_idx, best_path, best_pos, best_score = bp_idx, _path, pos, test_score
                            # do the exchange
            if not best_score == 999999:  # found an insertion location indeed
                # TODO check if change is made inplace
                cur_nop[best_path_idx] = best_path[:best_pos] + [node_j] + best_path[best_pos:]
            else:
                # construct new path into cur_nop
                new_path = [path_op[0], node_j, path_op[-1]]
                cur_nop.append(new_path)

        # pick up best from both path_op and path_nop
        solutions = [cur_op] + cur_nop
        # DEBUG
        best_score, best_path, best_index = -999999, [], -999999
        for index, solution in enumerate(solutions):
            if len(set(solution[1:-1])) < len(solution[1:-1]):
                raise LookupError('Duplicate nodes in a path')
            cur_score = self.eval_util(solution, pref)
            if cur_score > best_score:
                best_path, best_score, best_index = solution, cur_score, index
        p_op = solutions.pop(best_index)
        p_nop = solutions
        return p_op, p_nop

    def one_point_movement(self, path_op, path_nop, tmax, pref, deviation, record):
        # calculate points that are within ellipse
        o, d = path_op[0], path_op[-1]

        distance = []
        for _node in range(self.NodeNum):
            distance.append(self.timematrix[o][_node] + self.timematrix[_node][d] + self.dwellarray[_node])
        # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
        index = np.argsort(distance)
        # check time limitation
        available_nodes = [x for x in index if distance[x] <= tmax]
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
            best_path_index, best_pos, best_score = -999999, -1, -999999
            for path_index, _path in enumerate(paths):
                for pos in range(1, len(_path)):
                    test_path = _path[:pos] + [_node] + _path[pos:]
                    # check feasibility:
                    if self.time_callback(test_path) < tmax:
                        test_score = self.eval_util(test_path, pref)
                        # check total score increase
                        # if test_score > eval_util(_path):  # TODO total score here 是指每个path的还是record的？
                        if test_score > record:
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
                if best_score >= record - deviation:
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
            score.append(self.eval_util(_path, pref))
        path_op = paths.pop(np.argsort(score)[-1])
        return path_op, paths

    def two_opt(self, path_op, pref):
        best = list(path_op)
        _score = self.eval_util(best, pref)
        improved = True
        while improved:
            improved = False
            for _i in range(1, len(path_op) - 2):
                for j in range(_i + 1, len(path_op)):
                    if j - _i == 1:
                        continue
                    if self.cost_change(best[_i - 1], best[_i], best[j - 1], best[j]) < -0.001:
                        best[_i:j] = best[j - 1:_i - 1:-1]
                        improved = True
        return best if self.eval_util(best, pref) > _score else path_op  # check improvement of utility

    def reinitialization(self, path_op, path_nop, k, tmax, pref):
        if k < 1 or k > len(path_op) - 2:
            return path_op, path_nop
        ratio = []
        # visited = path_op[1:-1]
        for _idx in range(1, len(path_op) - 1):
            # si/costi
            gain = self.node_util_callback(path_op[_idx], pref, np.zeros([3]))
            cost = self.timematrix[path_op[_idx - 1]][path_op[_idx]] + self.dwellarray[path_op[_idx]] + \
                   self.timematrix[path_op[_idx]][
                       path_op[_idx + 1]] - self.timematrix[path_op[_idx - 1]][path_op[_idx + 1]]
            ratio.append(gain / cost)
        # ratio is benefit/insertion cost
        nodes_sorted = np.argsort(ratio)  # smaller nodes are assigned in front

        remove_indices = nodes_sorted[:k]
        # for _i, _node in enumerate(path_op):
        path_op_new = [path_op[x] for x in range(len(path_op)) if x - 1 not in remove_indices]
        for _k in range(k):
            remove_node_idx = remove_indices[_k] + 1
            # remove node from path_op
            node_j = path_op[remove_node_idx]  # TODO 每次path_op都在变小，不能按idx pop
            # put node_j back into path_nop
            # criteria: minimum insertion cost
            best_path_idx, best_path, best_pos, best_score = 999999, [], -1, 999999
            for bp_idx, _path in enumerate(path_nop):
                if node_j in _path[1:-1]:  # skip nodes that serve as origin or destination
                    raise LookupError('Duplicate nodes are not supposed to present! Debug please.')
                    # continue  # avoid repetitive existence
                for pos in range(1, len(_path)):
                    length = self.time_callback(_path)
                    # feasibility check
                    if self.timematrix[_path[pos - 1]][node_j] + self.dwellarray[node_j] + self.timematrix[node_j][
                        _path[pos]] - \
                            self.timematrix[_path[pos - 1]][_path[pos]] + length < tmax:
                        test_path = _path[:pos] + [node_j] + _path[pos:]
                        test_score = self.time_callback(test_path) - length
                        # find best insert position
                        if test_score <= best_score:
                            best_path_idx, best_path, best_pos, best_score = bp_idx, _path, pos, test_score
                            # do the exchange
            if not best_score == 999999:  # found an insertion location indeed
                path_nop[best_path_idx] = best_path[:best_pos] + [node_j] + best_path[best_pos:]
            else:
                # construct new path into cur_nop
                new_path = [path_op[0], node_j, path_op[-1]]
                path_nop.append(new_path)
        path_op = path_op_new
        return path_op, path_nop

    """
    path simularity evaluation function not complete
    """

    def path_penalty(self, Pa, Pb, distance_matrix):
        # path_a and path_b both starts from 0. offset starts from 0, i.e., 0 --> 1st destination
        if Pa[0] != Pb[0] or Pa[-1] != Pb[-1]:
            raise ValueError('Paths have different o or d.')
        # define insertion cost
        o, d = Pa[0], Pa[-1]
        insertion_cost = [distance_matrix[o][_] + distance_matrix[_][d] for _ in range(distance_matrix.shape[0])]

        # check empty path
        path_a, path_b = Pa[1:-1], Pb[1:-1]
        if not path_a or not path_b:
            if not path_a and not path_b:  # if all empty
                return 0
            elif path_a:  # path b is empty
                return sum([max(distance_matrix[x]) for x in path_a])  # calculate most distant results
            else:  # path a is empty
                return sum([max(distance_matrix[x]) for x in path_b])  # calculate most distant results
        # if both paths are not empty (with o, d excluded)
        # check node indices
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


if __name__ == '__main__':
    # %% Solver Setup
    NodeNum = 37  # number of attractions. Origin and destination are excluded.

    alpha1, alpha2 = -0.05, -0.05
    beta1, beta2, beta3 = 1, 0.03, 0.08  # TODO beta2该怎么定
    phi = 0.1

    Tmax = 500  # person specific time constraints
    Origin, Destination = 0, 0

    # %% save data
    # pickle.dump(UtilMatrix, open('UtilMatrix.txt', 'wb'))
    # pickle.dump(TimeMatrix, open('TimeMatrix.txt', 'wb'))
    # pickle.dump(CostMatrix, open('CostMatrix.txt', 'wb'))
    # pickle.dump(DwellArray, open('DwellArray.txt', 'wb'))

    UtilMatrix = pickle.load(open('Database/solver example/UtilMatrix.txt', 'rb'))
    TimeMatrix = pickle.load(open('Database/solver example/TimeMatrix.txt', 'rb'))
    TimeMatrix = (TimeMatrix + TimeMatrix.T) / 2

    CostMatrix = pickle.load(open('Database/solver example/CostMatrix.txt', 'rb'))
    DwellArray = pickle.load(open('Database/solver example/DwellArray.txt', 'rb'))

    solver = IlsUtility(NodeNum, [alpha1, alpha2], [beta1, beta2, beta3], phi, UtilMatrix, TimeMatrix, CostMatrix,
                        DwellArray)
    solver.modify_travel_time()

    # %% start solver
    # warning: total utility of a path must >= 0
    Pref = np.array([0.5, 0.3, 0.2])
    route = [0, 2, 4, 0]
    print('test %.2f \n' % solver.eval_util(route, Pref))

    # initialization
    PathOp, PathNop = solver.initialization(Tmax, Pref, Origin, Destination)

    print('Scores after initial insertion: \n')
    print('Optimal path score: {}, time: {}'.format(solver.eval_util(PathOp, Pref), solver.time_callback(PathOp)))
    print(PathOp)
    for i in PathNop:
        print('Non-optimal path score: {}, time: {}'.format(solver.eval_util(i, Pref), solver.time_callback(i)))
        print(i)

    record, p = solver.eval_util(PathOp, Pref), 0.15
    deviation = p * record
    best_solution = PathOp.copy()
    K = 3

    for _K in range(K):
        print('\nCurrent K loop number: {}'.format(_K))
        for itr in range(4):
            print('\nCurrent iteration: {}'.format(itr))
            # two-point exchange
            Path_op, Path_nop = solver.two_point_exchange(PathOp, PathNop, Tmax, Pref, record, deviation)
            visited = []
            print('\nScores after two-point exchange: \n')
            score = solver.eval_util(Path_op, Pref)
            print('Optimal path score: {}, time: {}'.format(score, solver.time_callback(Path_op)))
            print(Path_op)
            visited.extend(Path_op[1:-1])

            for i, path in enumerate(Path_nop):
                visited.extend(path[1:-1])
                print('Current path number: {}, score as {}, time: {}'.format(i, solver.eval_util(path, Pref),
                                                                              solver.time_callback(path)))
                print(path)

            print('Number of attractions visited: {}, duplicate nodes: {}.'.format(len(visited),
                                                                                   len(visited) - len(set(visited))))
            if score > record:
                best_solution, record = list(Path_op), score
                deviation = p * record

            # one-point movement
            Path_op, Path_nop = solver.one_point_movement(Path_op, Path_nop, Tmax, Pref, deviation, record)
            visited = []

            print('\nScores after one-point movement: \n')
            score = solver.eval_util(Path_op, Pref)
            print('Optimal path score: {}, time: {}'.format(score, solver.time_callback(Path_op)))
            print(Path_op)
            visited.extend(Path_op[1:-1])

            if score > record:
                best_solution, record = list(Path_op), score
                deviation = p * record

            for i, path in enumerate(Path_nop):
                visited.extend(path[1:-1])
                print('Current path number: {}, score as {}, time: {}'.format(i, solver.eval_util(path, Pref),
                                                                              solver.time_callback(path)))
                print(path)

            print('Number of attractions visited: {}, duplicate nodes: {}.'.format(len(visited),
                                                                                   len(visited) - len(set(visited))))

            # 2-opt (clean-up)
            print('\nPath length before 2-opt: {}, with score: {}'.format(solver.time_callback(Path_op),
                                                                          solver.eval_util(Path_op, Pref)))
            Path_op_2 = solver.two_opt(Path_op, Pref)
            cost_2_opt = solver.eval_util(Path_op_2, Pref)
            print('Path length after 2-opt: {},  with score: {}'.format(solver.time_callback(Path_op_2), cost_2_opt))

            PathOp, PathNop = Path_op_2, Path_nop

            # if no movement has been made, end I loop
            if Path_op_2 == best_solution:
                break
            # if a new better solution has been obtained, then set new record and new deviation
            if cost_2_opt > record:
                best_solution, record = list(Path_op_2), cost_2_opt
                deviation = p * record
        # perform reinitialization
        PathOp, PathNop = solver.reinitialization(PathOp, PathNop, 3, Tmax, Pref)

    print('\nBest solution score: {}, time: {} \nSolution: {}'.format(record, solver.time_callback(best_solution),
                                                                      best_solution))
