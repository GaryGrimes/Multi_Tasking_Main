import numpy as np
from ILS_master import IlsUtility
import multiprocessing

"""last modified on Oct. 24th 16:29"""


class SolverUtility(object):
    @staticmethod
    def solver(q, index, node_num, agent_database, **kwargs):
        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], \
                                                                                             kwargs['beta'], kwargs[
                                                                                                 'phi'], kwargs[
                                                                                                 'util_matrix'], kwargs[
                                                                                                 'time_matrix'], kwargs[
                                                                                                 'cost_matrix'], kwargs[
                                                                                                 'dwell_matrix'], \
                                                                                             kwargs[
                                                                                                 'dist_matrix']

        solver = IlsUtility(node_num, alpha, beta, phi, util_matrix, time_matrix, cost_matrix, dwell_matrix)

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), multiprocessing.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []
            # agent property (time budget, origin, destination)
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            # initialization
            init_res = solver.initialization(t_max, pref, origin, destination)
            if init_res is None or [] in init_res:
                continue
            else:
                path_op, path_nop = init_res  # get value from a tuple

            path_pdt.append(path_op)

            # print('------  Scores after initial insertion: ------\n')
            # print('Optimal path score: %.2f, time: %d' % (solver.eval_util(PathOp, Pref), solver.time_callback(PathOp)))
            # print_path(PathOp)
            # for i in PathNop:
            #     print('Non-optimal path score: %.2f, time: %d' % (solver.eval_util(i, Pref), solver.time_callback(i)))
            #     print_path(i)

            # try different deviations
            for p in [0.05, 0.1, 0.15]:
                # print('\n------ Current deviation: {} ------'.format(p))
                record = solver.eval_util(path_op, pref)
                deviation = p * record
                best_solution = path_op.copy()

                k = 3
                for _K in range(k):
                    # print('\n------ Current K loop number: {} ------'.format(_K))
                    for itr in range(4):
                        # print('\n---- Current iteration: {} ----'.format(itr))
                        # two-point exchange
                        path_op, path_nop = solver.two_point_exchange(path_op, path_nop, t_max, pref, record, deviation)

                        # print('\nScores after two-point exchange: \n')
                        score = solver.eval_util(path_op, pref)
                        # print('Optimal path score: %.2f, time: %d' % (score, solver.time_callback(Path_op)))
                        # print_path(Path_op)

                        if score > record:
                            path_pdt.append(path_op)
                            best_solution, record = list(path_op), score
                            deviation = p * record

                        # one-point movement
                        path_op, path_nop = solver.one_point_movement(path_op, path_nop, t_max, pref, deviation, record)
                        visited = []

                        # print('\nScores after one-point movement: \n')
                        score = solver.eval_util(path_op, pref)
                        # print('Optimal path score: %.2f, time: %d' % (score, solver.time_callback(Path_op)))
                        # print_path(Path_op)
                        visited.extend(path_op[1:-1])

                        if score > record:
                            path_pdt.append(path_op)
                            best_solution, record = list(path_op), score
                            deviation = p * record

                        # 2-opt (clean-up)
                        # print('\nPath length before 2-opt: %d, with score: %.2f' % (solver.time_callback(Path_op),
                        #                                                             solver.eval_util(Path_op, Pref)))
                        path_op_2 = solver.two_opt(path_op, pref)
                        cost_2_opt = solver.eval_util(path_op_2, pref)
                        # print('Path length after 2-opt: %d,  with score: %.2f' % (
                        # solver.time_callback(Path_op_2), cost_2_opt))

                        path_op, path_nop = path_op_2, path_nop

                        # if no movement has been made, end I loop
                        if path_op_2 == best_solution:
                            break
                        # if a new better solution has been obtained, then set new record and new deviation
                        if cost_2_opt > record:
                            path_pdt.append(path_op)
                            best_solution, record = list(path_op_2), cost_2_opt
                            deviation = p * record
                    # perform reinitialization
                    path_op, path_nop = solver.reinitialization(path_op, path_nop, _K, t_max, pref)

                # print('\nBest solution score: %.2f, time: %d ' % (record, solver.time_callback(best_solution)))
                # print_path(best_solution)
                path_pdt.append(best_solution)

            path_obs = list(np.array(_agent.path_obs) - 1)  # attraction indices start from 0

            # last modified on Oct. 24 16:29 2019

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(solver.eval_util(_path, pref))  # a list of penalties

                filter_ratio = 0.15  # predicted paths with penalties within 15% interval
                max_score = max(path_pdt_score)  # max utility score for current path

                ''' within 90% score or at least 3 paths in the predicted path set'''
                threshold = max_score - abs(filter_ratio * max_score)

                for _ in np.argsort(path_pdt_score)[::-1]:
                    if path_pdt_score[_] >= threshold:
                        selected_path.append(path_pdt[_])
                    else:
                        break
                # at least 3 paths in the set
                if len(selected_path) < 3:
                    selected_path = []
                    for _ in np.argsort(path_pdt_score)[-3:]:
                        selected_path.append(path_pdt[_])

            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = solver.distance_penalty(path_obs, _path, dist_matrix)
                if not best_path_predicted:
                    best_path_predicted, lowest_penalty = _path, res
                if res < lowest_penalty:
                    best_path_predicted, lowest_penalty = _path, res
                # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref).
                # solver.time_callback(_path)))
                # print('Penalty: {}'.format(res))
                # print_path(_path)

            # WRITE PREDICTED PATH AND PENALTY
            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        q.put((index, data))
