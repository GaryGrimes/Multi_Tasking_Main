import numpy as np
import numdifftools as nd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy
import slvr.PntyEvalFun_Batch as pf


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
    return - x ** 2 - y ** 2 + 3


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


# # 新建一个画布
# figure = plt.figure()
# # 新建一个3d绘图对象
# ax = Axes3D(figure)
#
# # 生成x, y 的坐标集 (-2,2) 区间，间隔为 0.1
# x = np.arange(-2, 2, 0.1)
# y = np.arange(-2, 2, 0.1)
#
# # 生成网格矩阵
# X, Y = np.meshgrid(x, y)
#
# # Z 轴 函数
# Z = - np.power(X, 2) - np.power(Y, 2) + 3
#
# # 定义x,y 轴名称
# plt.xlabel("x")
# plt.ylabel("y")
#
# # 设置间隔和颜色
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
# # 展示
# plt.show()

if __name__ == '__main__':

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

    # numerical gradient using * parameter
    s = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]

    res_Grad = nd.Gradient(pf.eval_fun)(s)
    res_Hessian = nd.Hessian(pf.eval_fun)(s)


    # # res_test = pf.eval_fun(s)
    # # print('The score of parameter of {}: {}'.format(s, res_test))
    # res_test = pf.eval_fun([0, 0, 0, 0])

    # test by making changes
    
    print('This is a test for modifying code from Github')
    
    
