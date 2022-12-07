from math import sqrt


def inf_norm(M, k):
    amax = 0.0
    for i in range(k):
        a = 0.0
        for j in range(k):
            a += abs(M[i, j])
        if a > amax:
            amax = a

    return amax


def vec_norm(v):

    return sqrt(sum(x*x for x in v))
