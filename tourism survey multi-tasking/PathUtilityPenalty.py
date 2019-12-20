def path_penalty_utility(self, Pa, Pb, distance_matrix):  # created on Nov.3 2019
    """Pa is observed path, Pb predicted"""

    # Offset is 0 for the 1st destination.

    if Pa[0] != Pb[0] or Pa[-1] != Pb[-1]:
        raise ValueError('Paths have different o or d.')

    # define the penalty in utility form for every two destinations. u_ik stands for the generalized cost of travel
    o, d = Pa[0], Pa[-1]

    path_a, path_b = Pa[1:-1], Pb[1:-1]  # excluding origin and destination

    # utility (negative) penalty evaluation
    cost, a, b = 0, o, o  # let a, b be origin

    # if exist empty path
    if not path_a:  # if observed path is empty
        return cost

    while path_a and path_b:
        a, b = path_a.pop(0), path_b.pop(0)  # a, b correspond to the i_th node in path_a, path_b
        cost += self.arc_util_callback(a, b)

    if path_a:  # length of path_a > path b
        while path_a:
            a = path_a.pop(0)
            cost += self.arc_util_callback(a, b)
    else:  # case when length of path_b > path a
        while path_b:
            b = path_b.pop(0)
            cost += self.arc_util_callback(a, b)
    return cost
