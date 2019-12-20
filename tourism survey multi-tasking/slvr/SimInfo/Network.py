""" Network database
With network attributes, travel detail matrices and methods"""

# Node attributes
node_num = 0
util_mat = []  # utility of attractions  node_num * 3
dwell_vec = []  # dwell time vector

# Edge attributes in numpy.ndarray
time_mat = []  # travel time matrix, node_num * node_num. Unit = minute.
cost_mat = []  # fare matrix
dist_mat = []


class Node(object):
    def __init__(self, dwell_time, int_utility, _id):
        self.dwell = dwell_time
        self.int_util = int_utility
        self.id = _id
        self.start_time, self.depart_time = 0, 0


class Edge(object):
    def __init__(self, global_phi, _origin, _destination, _time, cost, dist):
        self.phi = global_phi
        self.origin, self.destination = _origin, _destination
        self.travel_time = _time
        self.travel_cost = cost
        self.travel_distance = dist
        self.visit = 0

    def edge_util_callback(self, alpha):
        alpha_1, alpha_2, _phi = alpha[0], alpha[1], self.phi
        return alpha_1 * self.travel_time + alpha_2 * _phi * self.travel_cost


def modify_travel_time():
    global time_mat
    for i in range(time_mat.shape[0]):
        for j in range(time_mat.shape[1]):
            min_cost = time_mat[i][j]
            for k in range((time_mat.shape[1])):
                cost = time_mat[i][k] + time_mat[k][j]
                # if cost < min_cost:
                #     print('travel time error　from {} to {}.'.format(i, j))
                min_cost = cost if cost < min_cost else min_cost
            time_mat[i][j] = min(min_cost, time_mat[i][j])
    time_mat = (time_mat + time_mat.T) / 2


def travel_time_check():  # Edge travel time
    global time_mat
    # need several iterations to make sure direct travel is shorter than any detour
    no_update, itr = 0, 0
    print('Starting travel_time_check...')
    for _ in range(3):
        while not no_update:
            print('Current iteration: {}'.format(itr + 1))
            no_update = 1
            for i in range(time_mat.shape[0] - 1):
                for j in range(i + 1, time_mat.shape[0]):
                    time = time_mat[i, j]
                    shortest_node, shortest_time = 0, time
                    for k in range(time_mat.shape[0]):
                        if time_mat[i, k] + time_mat[k, j] < shortest_time:
                            shortest_node, shortest_time = k, time_mat[i, k] + time_mat[k, j]
                    if shortest_time < time:
                        no_update = 0
                        # print('travel time error between {0} and {1}, \
                        # shortest path is {0}-{2}-{1}'.format(i, j, shortest_node))
                        time_mat[j, i] = time_mat[i, j] = shortest_time
            itr += 1
            if no_update:
                print('Travel time update complete.\n')


if __name__ == '__main__':
    pass
