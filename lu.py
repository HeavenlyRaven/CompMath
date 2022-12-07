import numpy as np
import numpy.random as npr
import numpy.linalg as npl

from utils import inf_norm

# set number of digits after decimal point
# np.set_printoptions(precision=2, suppress=True)

# generating some random stuff
n = npr.randint(3, 10)
A = npr.rand(n, n)*20
b = npr.rand(n, 1)*10

# generating a singular matrix
S = npr.rand(n, n)*20
for i in range(npr.randint(1, n//2+1)):
    S[npr.randint(0, n)] = 0.0
for i in range(npr.randint(1, n//2+1)):
    S[npr.randint(0, n//2)] = S[npr.randint(n//2, n)]*npr.randint(1, 100)
for i in range(npr.randint(1, n//2+1)):
    S[npr.randint(0, n//2)] = S[npr.randint(n//2, n)]+S[npr.randint(n//2, n)]

# for system with singular matrix with solution:
B = np.copy(A)
c = np.copy(b)
B[n-1], c[n-1, 0] = B[n-2], c[n-2, 0]
B[0] = c[0, 0] = 0.0

# precision
eps = inf_norm(S, n)*1e-15

# number of arithmetic operations
op = 0


# perform LU = PAQ decomposition
def LU(A, n, mode="default"):

    global op

    LU = np.copy(A)

    # list representation of the matrix of row permutation
    p = [x for x in range(n)]
    # list representation of the matrix of column permutation
    q = [x for x in range(n)]
    # permutation flag
    c = 1
    # matrix rank
    rank = n

    # return indices of the pivot on k-th step in permutation lists
    def pivot(k):
        abs_piv = 0.0
        pi, qj = 0, 0
        for i in range(k, n):
            for j in range(k, n):
                abs_val = abs(LU[p[i], q[j]])
                if abs_val >= abs_piv:
                    abs_piv = abs_val
                    pi, qj = i, j
        return pi, qj

    # return matrices of row and column permutations
    def PnQ():

        return np.identity(n)[p], np.identity(n)[q].T

    # return L and U matrices
    def LnU():
        L = np.zeros((n, n))
        U = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i > j:
                    L[i, j] = LU[p[i], q[j]]
                else:
                    U[i, j] = LU[p[i], q[j]]
                    if i == j:
                        L[i, j] = 1.

        return L, U

    # actual algorithm
    for k in range(n):
        pi, qj = pivot(k)
        piv = LU[p[pi], q[qj]]
        if abs(piv) < eps:
            rank = k
            break
        if pi != k:
            p[k], p[pi] = p[pi], p[k]
            c = -c
        if qj != k:
            q[k], q[qj] = q[qj], q[k]
            c = -c
        for i in range(k+1, n):
            a = LU[p[i], q[k]]/piv
            op += 1
            LU[p[i], q[k]] = a
            for j in range(k+1, n):
                op += 2
                LU[p[i], q[j]] = LU[p[i], q[j]] - LU[p[k], q[j]]*a

    if mode == "default":
        return LnU(), PnQ()
    elif mode == "det":
        return LU, p, q, c, rank
    elif mode == "solve":
        return LU, p, q, rank


def determinant(A, n):

    LUdata, p, q, c, rank = LU(A, n, mode="det")

    if rank < n:
        return 0
    else:
        det = c
        for i in range(n):
            det *= LUdata[p[i], q[i]]
        return det


def solve(LU, p, q, rank, n, b):

    global op

    x = np.zeros((n, 1))
    y = [0.0]*n
    z = [0.0]*n

    # solve Ly = Pb
    for i in range(n):
        s = b[p[i], 0]
        for j in range(i):
            op += 2
            s -= LU[p[i], q[j]]*y[j]
        y[i] = s

    for i in range(rank, n):
        if y[i] != 0.0:
            raise Exception("No solution")
        else:
            y[i] = 1.

    # solve Uz = y
    for i in range(n-rank, n):
        s = y[n-i-1]
        for j in range(i):
            op += 2
            s -= LU[p[n-i-1], q[n-j-1]]*z[n-j-1]
        op += 1
        z[n-i-1] = s/LU[p[n-i-1], q[n-i-1]]

    # solve x = Qz
    for i in range(n):
        x[q[i], 0] = z[i]

    return x


def inverse(A, n):

    LUdata = LU(A, n, mode="solve")
    ainv = []
    for i in range(n):
        e = np.zeros((n, 1))
        e[i, 0] = 1
        ainv.append(solve(*LUdata, n, e))

    return np.column_stack(ainv)


def condition_number(A, n, norm="inf"):

    if norm == "inf":
        return inf_norm(A, n)*inf_norm(inverse(A, n), n)


#(L, U), (P, Q) = LU(A, n)
# print(P @ A @ Q, end='\n\n')
#print(L)
#print(U)
#print(np.allclose(P @ A @ Q, L @ U))
# print(determinant(A, n))
# print(npl.det(A))
#x = solve(*LU(A, n, mode="solve"), n, b)
# Ainv = inverse(A, n)
# print(np.allclose(Ainv @ b, x))
# print(A @ Ainv)
# print(Ainv @ A)
# print(condition_number(A, n))
# print(npl.cond(A, p=np.inf))
#x = solve(*LU(B, n, mode="solve"), n, c)
#print(np.allclose(B @ x, c))
# print(npl.matrix_rank(S))










