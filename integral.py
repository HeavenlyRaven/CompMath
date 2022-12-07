from math import cos, sin, exp, log, ceil, floor
import numpy as np
# from scipy.integrate import quad

import qr

# parameters
a = 1.1
b = 2.5
alpha = 2/5
# beta = 0.0


# function f(x)
def f(x):
    return 0.5*cos(2*x)*exp(2*x/5)+2.4*sin(1.5*x)*exp(-6*x)+6*x


# weight function p(x)
def p(x):

    return (x-1.1)**(-0.4)


# exact value
# J = quad(lambda x: f(x)*p(x), a, b)[0]
J = 18.60294785731848208626949366919856494853

# Mn = 13
# M2n = 150
# Rn <= 0.35
# R2n <= 0.001


# auxiliary function for calculating moments
def I(k, z1, z2):
    return ((z2 - a) ** (k+1-alpha) - (z1 - a) ** (k+1-alpha)) / (k+1-alpha)


# Newton-Cotes rule
def newton_cotes(f, z1, z2):

    z12 = (z1+z2)/2

    mu0 = I(0, z1, z2)
    mu1 = I(1, z1, z2) + a*mu0
    mu2 = I(2, z1, z2) + 2*a*mu1 - a*a*mu0

    A1 = (mu2 - mu1*(z12 + z2) + mu0*z12*z2)/((z12-z1)*(z2-z1))
    A2 = -(mu2 - mu1*(z1 + z2) + mu0*z1*z2)/((z12-z1)*(z2-z12))
    A3 = (mu2 - mu1*(z12 + z1) + mu0*z12*z1)/((z2-z12)*(z2-z1))

    return A1*f(z1)+A2*f(z12)+A3*f(z2)


# Gauss rule
def gauss(f, z1, z2):

    a2 = a*a
    a3 = a2*a
    a4 = a3*a
    a5 = a4*a

    mu0 = I(0, z1, z2)
    mu1 = I(1, z1, z2) + a*mu0
    mu2 = I(2, z1, z2) + 2*a*mu1 - a2*mu0
    mu3 = I(3, z1, z2) + 3*a*mu2 - 3*a2*mu1 + a3*mu0
    mu4 = I(4, z1, z2) + 4*a*mu3 - 6*a2*mu2 + 4*a3*mu1 - a4*mu0
    mu5 = I(5, z1, z2) + 5*a*mu4 - 10*a2*mu3 + 10*a3*mu2 - 5*a4*mu1 + a5*mu0

    Mu = np.array([
        [mu0, mu1, mu2],
        [mu1, mu2, mu3],
        [mu2, mu3, mu4]
    ])
    mup = np.array([[-mu3], [-mu4], [-mu5]])
    p = qr.solve(Mu, 3, mup)

    x = np.roots((1., p[2, 0], p[1, 0], p[0, 0]))
    if any((r > z2 or r < z1 for r in x)):
        raise Exception("Roots out of bounds")
    x1, x2, x3 = x[2], x[1], x[0]

    X = np.array([[1., 1., 1.], [x1, x2, x3], [x1*x1, x2*x2, x3*x3]])
    mua = np.array([[mu0], [mu1], [mu2]])
    A = qr.solve(X, 3, mua)

    return A[0, 0]*f(x1)+A[1, 0]*f(x2)+A[2, 0]*f(x3)


# simple compound quadrature rule
def simple_compound(f, h, rule):
    if h > b-a:
        raise Exception("Step too large")
    z1 = a
    z2 = a + h
    S = 0.0
    for i in range(int((b - a) / h)):
        S += rule(f, z1, z2)
        z1 = z2
        z2 += h
    return S


def compound(f, rule, h=b-a, L=2, eps=1e-6):

    ITER_LIMIT = 1000
    Se = [0.0]*3
    print(h)

    for i in range(ITER_LIMIT):
        S = simple_compound(f, h, rule)
        print((b-a)/h)
        Se[0] = Se[1]
        Se[1] = Se[2]
        Se[2] = S
        if (i+1) >= 3:
            m = -log(abs((Se[2]-Se[1])/(Se[1]-Se[0])))/log(L)
            print(m)
            if (Se[2]-Se[1])/(L**m-1) < eps:
                return S
        h /= L


def compound_opt(f, rule, L=2, eps=1e-6, fac=0.95):

    h = b-a
    Sh1 = simple_compound(f, h, rule)
    Sh2 = simple_compound(f, h/L, rule)
    Sh3 = simple_compound(f, h/(L*L), rule)

    m = -log((Sh3 - Sh2) / (Sh2 - Sh1)) / log(L)

    hopt = fac*h*((eps*(1-L**(-m)))/abs(Sh2-Sh1))**(1/m)
    hopt = (b-a)/floor((b-a)/4/hopt)
    print((b-a)/hopt)

    return compound(f, rule, h=hopt, L=L, eps=eps)

# print(J)
print(J-compound(f, newton_cotes))