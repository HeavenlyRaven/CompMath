import numpy as np
import numpy.random as npr
import numpy.linalg as npl

from math import log
from utils import inf_norm, vec_norm

# set number of digits after decimal point
np.set_printoptions(precision=2, suppress=True)

# generating some random stuff
n = npr.randint(3, 10)
# A = npr.rand(n, n)*20
b = npr.rand(n, 1)*20

# generate diagonally dominant matrix
D = npr.rand(n, n)*20
for i in range(n):
    s = 0
    for j in range(n):
        s += abs(D[i, j])
    D[i, i] = s*2

# generate symmetric positive-definite matrix
S = npr.rand(n, n)*2
S = S @ S.T
S = (S + S.T)/2


def SIM(A, b, n, eps):

    B, c = jacobi(A, b, n)

    x0 = np.copy(c)
    x1 = np.zeros((n, 1))
    q = inf_norm(B, n)

    # a priori estimate
    k1 = int(log(eps*(1-q)/vec_norm(c), q))
    # a posteriori estimate
    k2 = k1

    e = eps*(1/q-1)

    for i in range(k1):
        x1 = B @ x0 + c
        if vec_norm(x1-x0) <= e:
            k2 = i+1
            break
        x0 = np.copy(x1)

    print(f"a priori: {k1}, a posteriori: {k2}")
    return x1


def jacobi(A, b, n):

    B = np.copy(A)
    c = np.copy(b)

    for i in range(n):
        a = A[i, i]
        B[i] /= -a
        B[i, i] = 0.0
        c[i, 0] /= a

    return B, c


def seidel(A, b, n, eps, mode="DDM", ITER_LIMIT=10000):

    B, c = jacobi(A, b, n)

    BL = np.zeros((n, n))
    BR = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i > j:
                BL[i, j] = B[i, j]
            else:
                BR[i, j] = B[i, j]

    x0 = np.copy(c)
    x1 = np.zeros((n, 1))

    # a posteriori estimate
    k2 = None

    e = eps*(1-inf_norm(B, n))/inf_norm(BR, n) if mode == "DDM" else eps

    for k in range(ITER_LIMIT):
        for i in range(n):
            x1[i] = BL[i] @ x1 + BR[i] @ x0 + c[i]
        if vec_norm(x1 - x0) <= e:
            k2 = k
            break
        x0 = np.copy(x1)

    print(f"a posteriori: {k2}")
    return x1


x = seidel(D, b, n, 1e-16)
print(np.allclose(D @ x, b))
# print(npl.solve(D, b))
