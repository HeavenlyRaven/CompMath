from math import sqrt
import numpy as np
# import numpy.random as npr
# import numpy.linalg as npl

# set number of digits after decimal point
# np.set_printoptions(precision=2, suppress=True)


# perform A = QR decomposition
def QR(A, n):

    R = np.copy(A)
    Q = np.identity(n)

    def givens(a, b):

        r = sqrt(a*a+b*b)
        return a/r, b/r

    for j in range(n):
        for i in range(n-1, j, -1):
            c, s = givens(R[i-1, j], R[i, j])
            for k in range(n):
                r1, r2 = R[i-1, k], R[i, k]
                R[i-1, k] = c*r1 + s*r2
                R[i, k] = -s*r1 + c*r2
                q1, q2 = Q[k, i-1], Q[k, i]
                Q[k, i-1] = c*q1 + s*q2
                Q[k, i] = -s*q1 + c*q2

    return Q, R


def solve(A, n, b):

    Q, R = QR(A, n)

    x = np.zeros((n, 1))
    y = [0]*n

    # solve Qy = b
    for i in range(n):
        s = 0
        for j in range(n):
            s += Q[j, i]*b[j, 0]
        y[i] = s

    # solve Rx = y
    for i in range(n):
        s = y[n-i-1]
        for j in range(i):
            s -= R[n-i-1, n-j-1]*x[n-j-1, 0]
        x[n-i-1, 0] = s/R[n-i-1, n-i-1]

    return x


# s = 0
# o = 0
# t = 0
# N = 200
# for i in range(N):
    # n = npr.randint(3, 20)
    # A = npr.rand(n, n) * 20
    # b = npr.rand(n, 1) * 10

    # Q, R = QR(A, n)
    # if np.allclose(npl.inv(Q), Q.T):
         # o += 1
    # if np.allclose(A, Q @ R):
         # t += 1
    # if np.allclose(solve(A, n, b), npl.solve(A, b)):
         # s += 1



