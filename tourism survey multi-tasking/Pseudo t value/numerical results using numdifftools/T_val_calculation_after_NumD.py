import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import os
import pylab as pl
import itertools

if __name__ == '__main__':
    s = np.array([-1.286284872, -0.286449175, 0.691566901, 0.353739632])

    # Hessian evaluated using Numdifftools but the solver function didn't divide the penalty by average.
    # This is also a result with feasbile inverse and all positive standard errors
    raw_hess = pd.read_excel('Hessian at B star (no average).xlsx', header=None).iloc[1:, 1:]

    num_grad, num_hess = pd.read_excel('Gradient_Batch.xlsx', sheet_name='data', header=None), pd.read_excel(
        'Numerical Hessian Batch.xlsx', sheet_name='data', header=None)

    num_grad, num_hess = np.array(num_grad), np.array(num_hess)

    raw_hess = np.array(raw_hess)

    # ---------------------------- calculate pesudo t value  ------------------------- #

    variance = np.linalg.inv(-num_hess)
    std_err = np.sqrt(np.diag(variance))

    t_value = s / std_err

    # ---------------------------- print results ------------------------- #

    print('The numerical gradient matrix: \n {}\n'.format(num_grad))

    # print second derivatives
    print('The second derivative matrix:\n {}'.format(num_hess))
    print('The standard error:\n {}'.format(std_err))

    print('The t values for current beta*:\n {}\n'.format(t_value))

    # ---------------------------- calculate pesudo t value for raw result ------------------------- #
    '''	• 218 tourists with observed route but no preference… skipped in score evaluation. 
    1,511 - 218 = 1,293 tourists to evaluate for each iteration'''

    raw_hess = 1000 * raw_hess /1293

    variance1 = np.linalg.inv(raw_hess)
    std_err1 = np.sqrt(np.diag(variance1))

    t_value = s / std_err1

    # ---------------------------- print results ------------------------- #

    print('The second derivative matrix:\n {}'.format(raw_hess))
    print('The standard error:\n {}'.format(std_err1))

    print('The t values for current beta*:\n {}'.format(t_value))
