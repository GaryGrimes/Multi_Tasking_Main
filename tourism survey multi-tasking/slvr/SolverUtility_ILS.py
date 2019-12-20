""" This script contains main method for multi-tasking module, the TTDP solver.
Data Wrapping is executed in parent script to avoid repetitive I/Os"""

import numpy as np
import multiprocessing as mp
from SimInfo import Solver_ILS
import datetime

"""last modified on Nov. 16 17:23"""


class SolverUtility(object):
    @staticmethod
    def solver(q, process_idx, node_num, agent_database, **kwargs):  # levestain distance, with path threshold filter
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

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            # todo
            """ 如果要考虑multiple-day travel,则可以加一个for _ in range(_agent.day_of_travel)，
            每一天的predicted path加入visit_history"""
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                continue

            if len(initial_path) <= 2:
                final_order = initial_path
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

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

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
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

            # update progress bar

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        q.put((process_idx, data['penalty']))

        # return (process_idx, data)  # for debug

    @staticmethod
    def solver_num_Hes(q, process_idx, node_num, agent_database, **kwargs):  #
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

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            # todo
            """ 如果要考虑multiple-day travel,则可以加一个for _ in range(_agent.day_of_travel)，
            每一天的predicted path加入visit_history"""
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                continue

            if len(initial_path) <= 2:
                final_order = initial_path
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

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

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
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

        # para_penalty = sum(_penalty) / len(_penalty) if len(_penalty) > 0 else 0  # average error
        para_penalty = sum(_penalty) / 1293  # PT tourists

        data = {'penalty': para_penalty, 'predicted': predicted,
                'observed': observed}  # unit of penalty should be transformed from 'meter' to 'kilometer'.

        q.put((process_idx, data['penalty']))

        # return (process_idx, data)  # for debug

    @staticmethod
    def solver_debug(process_idx, node_num, agent_database, **kwargs):  #
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

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        # enumerate all tourists
        success_usr_cnt = 0
        success_set = set()
        initial_skip = 0

        # error type for unsuccessful tourists
        err_emty_info = []
        err_no_path = []
        err_init = []

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            # todo
            """ 如果要考虑multiple-day travel,则可以加一个for _ in range(_agent.day_of_travel)，
            每一天的predicted path加入visit_history"""
            visit_history = {}

            if pref is None or observed_path is None:
                err_emty_info.append(_idd)  # for debug
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                err_no_path.append(_idd)  # for debug
                continue

            start_time = datetime.datetime.now()

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            no_init_flag = 0
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                err_init.append(_idd)
                # initial_skip += 1
                # continue

            if len(initial_path) <= 2:
                no_init_flag = 1
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019
            # last modified on Dec. 20

            if no_init_flag:
                # do compulsory fill
                path_pdt.append(Solver_ILS.comp_fill())
                pass

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

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

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
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

            end_time = datetime.datetime.now()
            success_usr_cnt += 1
            success_set.add(_idd)
            t_passed = (end_time - start_time).seconds
            if t_passed > 60:
                print('------ Evaluation time: {}s for agent id {}------\n'.format(t_passed, _idd))
            # update progress bar

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        print('Successfully evaluated {} users. {} users were skipped, including {} of initial path.'.format(
            success_usr_cnt,
            len(agent_database) - success_usr_cnt, initial_skip))

        res_dict = {'process': process_idx,
                    'penalty': sum(_penalty) / 1000,
                    'initial skip': initial_skip,
                    'error_emty_info': err_emty_info,
                    'error_init': err_init,
                    'error_no_path': err_no_path}
        return res_dict  # for debug

    @staticmethod  # levestain distance, no path threshold filter
    def solver_LD_noPF(q, process_idx, node_num, agent_database, **kwargs):
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

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            # todo
            """ 如果要考虑multiple-day travel,则可以加一个for _ in range(_agent.day_of_travel)，
            每一天的predicted path加入visit_history"""
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                continue

            if len(initial_path) <= 2:
                final_order = initial_path
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019

            """对比的是combinatorial path score"""

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in path_pdt:
                res = Solver_ILS.path_penalty(path_obs, _path)  # path penalty (prediction error)
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

            # update progress bar

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        q.put((process_idx, data['penalty']))

    @staticmethod  # levestain distance, no path threshold filter
    def solver_single(node_num, agent, **kwargs):  # levestain distance, with path threshold filter
        '''the solver function is for single agent'''

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

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        pref = agent.preference
        observed_path = agent.path_obs
        t_max, origin, destination = agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

        visit_history = {}

        if pref is None or observed_path is None:
            raise ValueError('This agent cannot be evaluated')
        # skip empty paths (no visited location)
        if len(observed_path) < 3:
            raise ValueError('This agent cannot be evaluated')

        # ... since nodes controls visit history in path update for each agent
        Solver_ILS.node_setup(**node_properties)

        # agents setup
        agent_properties = {'time_budget': t_max,
                            'origin': origin,
                            'destination': destination,
                            'preference': pref,
                            'visited': visit_history}

        Solver_ILS.agent_setup(**agent_properties)

        # each path_op will be saved into the predicted path set for agent n
        path_pdt = []

        # %% strat up solver
        # solver initialization
        initial_path = Solver_ILS.initial_solution()

        # skip agents with empty initialized path
        if not initial_path:
            raise ValueError('This agent cannot be evaluated')

        if len(initial_path) <= 2:
            final_order = initial_path
        else:
            first_visit = initial_path[1]
            Solver_ILS.Node_list[first_visit].visit = 1

            order = initial_path
            final_order = list(order)

            # No edgeMethod in my case
            _u, _u8, _U10 = [], [], []

            counter_2 = 0
            no_improve = 0
            best_found = float('-inf')

            while no_improve < 50:
                best_score = float('-inf')
                local_optimum = 0

                # print(Order)

                while local_optimum == 0:
                    local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                counter_2 += 1  # 2指inner loop的counter
                v = len(order) - 1

                _u.append(best_score)  # TODO U is utility memo
                _u8.append(v)
                _U10.append(max(_u))

                if best_score > best_found:
                    best_found = best_score
                    final_order = list(order)

                    # save intermediate good paths into results
                    path_pdt.append(list(final_order))
                    no_improve = 0  # improved
                else:
                    no_improve += 1

                if len(order) <= 2:
                    continue
                else:
                    s = np.random.randint(1, len(order) - 1)
                    R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                if s >= min(_u8):
                    s = s - min(_u8) + 1

                order = Solver_ILS.shake(order, s, R)

        # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
        #                                                                           Solver_ILS.time_callback(
        #                                                                               final_order),
        #                                                                           Solver_ILS.eval_util(
        #                                                                               final_order)))

        # Prediction penalty evaluation. Compare the predicted paths with observed one.

        path_obs = list(
            np.array(agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

        # last modified on Oct. 24 16:29 2019

        """对比的是combinatorial path score"""
        selected_path = []
        if len(path_pdt) < 3:  # return directly if...
            selected_path = list(path_pdt)
        else:
            # evaluate scores for all path predicted (not penalty with the observed path here)
            path_pdt_score = []
            for _path in path_pdt:
                path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

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

        # -------- Path penalty evaluation --------
        # compare predicted path with observed ones
        # modified on Nov. 3 2019
        # last modified on Nov. 16

        best_path_predicted, lowest_penalty = [], float('inf')
        for _path in selected_path:
            res = Solver_ILS.path_penalty(path_obs, _path)
            if not best_path_predicted:
                best_path_predicted, lowest_penalty = _path, res
            if res < lowest_penalty:
                best_path_predicted, lowest_penalty = _path, res

        return lowest_penalty

    @staticmethod  # levestain distance, no path threshold filter
    def solver_debug_mp(q, process_idx, node_num, agent_database, **kwargs):  #
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

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        # enumerate all tourists
        success_usr_cnt = 0
        success_set = set()

        # error type for unsuccessful tourists
        err_emty_info = []
        err_no_path = []
        err_init = []

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            visit_history = {}

            if pref is None or observed_path is None:
                err_emty_info.append(_idd)  # for debug
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                err_no_path.append(_idd)  # for debug
                continue

            start_time = datetime.datetime.now()

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            no_init_flag = 0
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                err_init.append(_idd)
                # initial_skip += 1
                # continue

            if len(initial_path) <= 2:
                no_init_flag = 1
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019
            # last modified on Dec. 20

            if no_init_flag:
                # do compulsory fill
                path_pdt.append(Solver_ILS.comp_fill())
                pass

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

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

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
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

            end_time = datetime.datetime.now()
            success_usr_cnt += 1
            success_set.add(_idd)
            t_passed = (end_time - start_time).seconds
            if t_passed > 60:
                print('------ Evaluation time: {}s for agent id {}------\n'.format(t_passed, _idd))
            # update progress bar

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        print('For process {}, successfully evaluated {} users. {} users were skipped.'.
              format(process_idx, success_usr_cnt, len(agent_database) - success_usr_cnt, ))

        q.put((process_idx, data['penalty']))


if __name__ == '__main__':
    pass
