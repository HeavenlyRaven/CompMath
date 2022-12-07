import math
import numpy as np
import numpy.linalg as npl

from utils import vec_norm, inf_norm
from lu import LU, solve


ITER_LIMIT = 1000
eps = 1e-15

x0 = np.array([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
x0[4] = -0.2


def F(x):

    return np.mat([
        math.cos(x[1, 0] * x[0, 0]) - math.exp(-3 * x[2, 0]) + x[3, 0] * x[4, 0] ** 2 - x[5, 0] - math.sinh(2 * x[7, 0]) * x[8, 0] + 2 * x[9, 0] + 2.000433974165385440,
        math.sin(x[1, 0] * x[0, 0]) + x[2, 0] * x[8, 0] * x[6, 0] - math.exp(-x[9, 0] + x[5, 0]) + 3 * x[4, 0] ** 2 - x[5, 0] * (x[7, 0] + 1) + 10.886272036407019994,
        x[0, 0] - x[1, 0] + x[2, 0] - x[3, 0] + x[4, 0] - x[5, 0] + x[6, 0] - x[7, 0] + x[8, 0] - x[9, 0] - 3.1361904761904761904,
        2 * math.cos(-x[8, 0] + x[3, 0]) + x[4, 0] / (x[2, 0] + x[0, 0]) - math.sin(x[1, 0] ** 2) + math.cos(x[6, 0] * x[9, 0]) ** 2 - x[7, 0] - 0.1707472705022304757,
        math.sin(x[4, 0]) + 2 * x[7, 0] * (x[2, 0] + x[0, 0]) - math.exp(-x[6, 0] * (-x[9, 0] + x[5, 0])) + 2 * math.cos(x[1, 0]) - 1.0 / (-x[8, 0] + x[3, 0]) - 0.3685896273101277862,
        math.exp(x[0, 0] - x[3, 0] - x[8, 0]) + x[4, 0] ** 2 / x[7, 0] + math.cos(3 * x[9, 0] * x[1, 0]) / 2 - x[5, 0] * x[2, 0] + 2.0491086016771875115,
        x[1, 0] ** 3 * x[6, 0] - math.sin(x[9, 0] / x[4, 0] + x[7, 0]) + (x[0, 0] - x[5, 0]) * math.cos(x[3, 0]) + x[2, 0] - 0.7380430076202798014,
        x[4, 0] * (x[0, 0] - 2 * x[5, 0]) ** 2 - 2 * math.sin(-x[8, 0] + x[2, 0]) + 0.15e1 * x[3, 0] - math.exp(x[1, 0] * x[6, 0] + x[9, 0]) + 3.5668321989693809040,
        7 / x[5, 0] + math.exp(x[4, 0] + x[3, 0]) - 2 * x[1, 0] * x[7, 0] * x[9, 0] * x[6, 0] + 3 * x[8, 0] - 3 * x[0, 0] - 8.4394734508383257499,
        x[9, 0] * x[0, 0] + x[8, 0] * x[1, 0] - x[7, 0] * x[2, 0] + math.sin(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0] - 0.78238095238095238096]).T


def J(x):
    return np.mat([[-x[1, 0] * math.sin(x[1, 0] * x[0, 0]), -x[0, 0] * math.sin(x[1, 0] * x[0, 0]), 3 * math.exp(-3 * x[2, 0]), x[4, 0] ** 2, 2 * x[3, 0] * x[4, 0],
                    -1, 0, -2 * math.cosh(2 * x[7, 0]) * x[8, 0], -math.sinh(2 * x[7, 0]), 2],
                   [x[1, 0] * math.cos(x[1, 0] * x[0, 0]), x[0, 0] * math.cos(x[1, 0] * x[0, 0]), x[8, 0] * x[6, 0], 0, 6 * x[4, 0],
                    -math.exp(-x[9, 0] + x[5, 0]) - x[7, 0] - 1, x[2, 0] * x[8, 0], -x[5, 0], x[2, 0] * x[6, 0], math.exp(-x[9, 0] + x[5, 0])],
                   [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                   [-x[4, 0] / (x[2, 0] + x[0, 0]) ** 2, -2 * x[1, 0] * math.cos(x[1, 0] ** 2), -x[4, 0] / (x[2, 0] + x[0, 0]) ** 2, -2 * math.sin(-x[8, 0] + x[3, 0]),
                    1.0 / (x[2, 0] + x[0, 0]), 0, -2 * math.cos(x[6, 0] * x[9, 0]) * x[9, 0] * math.sin(x[6, 0] * x[9, 0]), -1,
                    2 * math.sin(-x[8, 0] + x[3, 0]), -2 * math.cos(x[6, 0] * x[9, 0]) * x[6, 0] * math.sin(x[6, 0] * x[9, 0])],
                   [2 * x[7, 0], -2 * math.sin(x[1, 0]), 2 * x[7, 0], 1.0 / (-x[8, 0] + x[3, 0]) ** 2, math.cos(x[4, 0]),
                    x[6, 0] * math.exp(-x[6, 0] * (-x[9, 0] + x[5, 0])), -(x[9, 0] - x[5, 0]) * math.exp(-x[6, 0] * (-x[9, 0] + x[5, 0])), 2 * x[2, 0] + 2 * x[0, 0],
                    -1.0 / (-x[8, 0] + x[3, 0]) ** 2, -x[6, 0] * math.exp(-x[6, 0] * (-x[9, 0] + x[5, 0]))],
                   [math.exp(x[0, 0] - x[3, 0] - x[8, 0]), -1.5 * x[9, 0] * math.sin(3 * x[9, 0] * x[1, 0]), -x[5, 0],-math.exp(x[0, 0] - x[3, 0] - x[8, 0]),
                    2 * x[4, 0] / x[7, 0], -x[2, 0], 0, -x[4, 0] ** 2 / x[7, 0] ** 2, -math.exp(x[0, 0] - x[3, 0] - x[8, 0]), -1.5 * x[1, 0] * math.sin(3 * x[9, 0] * x[1, 0])],
                   [math.cos(x[3, 0]), 3 * x[1, 0] ** 2 * x[6, 0], 1, -(x[0, 0] - x[5, 0]) * math.sin(x[3, 0]), x[9, 0] / x[4, 0] ** 2 * math.cos(x[9, 0] / x[4, 0] + x[7, 0]),
                    -math.cos(x[3, 0]), x[1, 0] ** 3, -math.cos(x[9, 0] / x[4, 0] + x[7, 0]), 0, -1.0 / x[4, 0] * math.cos(x[9, 0] / x[4, 0] + x[7, 0])],
                   [2 * x[4, 0] * (x[0, 0] - 2 * x[5, 0]), -x[6, 0] * math.exp(x[1, 0] * x[6, 0] + x[9, 0]), -2 * math.cos(-x[8, 0] + x[2, 0]), 1.5,
                   (x[0, 0] - 2 * x[5, 0]) ** 2, -4 * x[4, 0] * (x[0, 0] - 2 * x[5, 0]), -x[1, 0] * math.exp(x[1, 0] * x[6, 0] + x[9, 0]), 0, 2 * math.cos(-x[8, 0] + x[2, 0]),
                    -math.exp(x[1, 0] * x[6, 0] + x[9, 0])],
                   [-3, -2 * x[7, 0] * x[9, 0] * x[6, 0], 0, math.exp(x[4, 0] + x[3, 0]), math.exp(x[4, 0] + x[3, 0]),
                    -7.0 / x[5, 0] ** 2, -2 * x[1, 0] * x[7, 0] * x[9, 0], -2 * x[1, 0] * x[9, 0] * x[6, 0], 3, -2 * x[1, 0] * x[7, 0] * x[6, 0]],
                   [x[9, 0], x[8, 0], -x[7, 0], math.cos(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0], math.cos(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0],
                    math.cos(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0], math.sin(x[3, 0] + x[4, 0] + x[5, 0]), -x[2, 0], x[1, 0], x[0, 0]]])


def classic(F, J, x0):

    c = 0
    x = np.copy(x0)
    x_new = np.zeros((10, 1))
    for i in range(ITER_LIMIT):
        dx = solve(*LU(J(x), 10, mode="solve"), 10, -F(x))
        x_new = x + dx
        if vec_norm(x_new - x) <= eps:
            c = i + 1
            break
        x = np.copy(x_new)

    return x_new, c


def modified(F, J, x0):

    c = 0
    J0 = J(x0)
    LUdata = LU(J0, 10, mode="solve")
    x = np.copy(x0)
    x_new = np.zeros((10, 1))
    for i in range(ITER_LIMIT):
        dx = solve(*LUdata, 10, -F(x))
        x_new = x + dx
        if vec_norm(x_new - x) <= eps or vec_norm(F(x)) >= 1e+16:
            c = i+1
            break
        x = np.copy(x_new)

    return x_new, c


def classic_mod(F, J, x0, k):

    c = 0
    x = np.copy(x0)
    x_new = np.zeros((10, 1))
    LUdata = LU(J(x), 10, mode="solve")
    for i in range(ITER_LIMIT):
        if i <= k:
            LUdata = LU(J(x), 10, mode="solve")
        dx = solve(*LUdata, 10, -F(x))
        x_new = x + dx
        if vec_norm(x_new - x) <= eps:
            c += i+1
            break
        x = np.copy(x_new)
        if i > 990:
            print(x[0, 0])

    return x_new, c


def hybrid(F, J, x0, k):

    c = 0
    x = np.copy(x0)
    x_new = np.zeros((10, 1))
    LUdata = LU(J(x), 10, mode="solve")
    for i in range(ITER_LIMIT):
        if i % k == 0:
            LUdata = LU(J(x), 10, mode="solve")
        dx = solve(*LUdata, 10, -F(x))
        x_new = x + dx
        if vec_norm(x_new - x) <= eps:
            c = i + 1
            break
        x = np.copy(x_new)

    return x_new, c


x, c = classic_mod(F, J, x0, 4)
# x, c = hybrid(F, J, x0, 4)
print(x)
print(c)
print(F(x))
