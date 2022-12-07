import math
# заданное значение оценки погрешности функции
eps = 1e-6
# интересующие нас значения аргумента
interval = (0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2)


def sqrt(a: float) -> float:

    if a <= 1:
        x_prev = 1
    else:
        x_prev = a

    x_next = 0.5*(x_prev + a/x_prev)
    # ранее вычисленное значение оценки методической погрешности
    dt = eps/16.98

    # итерационная формула Герона
    while abs(x_next-x_prev) > dt:
        x_prev = x_next
        x_next = 0.5*(x_prev + a/x_prev)

    return x_next


def cosh(x: float) -> float:

    du = eps/1.53

    ux = 1
    s = 0
    i = 2

    while abs(2*ux/3) > du:
        s += ux
        ux = ux*x*x/((i-1)*i)
        i += 2

    return s


def cos(x: float) -> float:

    # TODO: Разобраться, насколько эффективно данное приведение
    sin = 0
    mf = 1
    x = abs(x)
    if x >= 2*math.pi:
        x = x % (2*math.pi)
    if x >= math.pi:
        x = 2*math.pi - x
    if x >= math.pi/2:
        mf = -1
        x = math.pi - x
    if x >= math.pi/4:
        sin = 1
        x = math.pi/2 - x

    dv = eps/12.39

    ux = x**sin
    s = 0
    i = 2+sin

    while abs(ux) > dv:
        s += ux
        ux = -ux*x*x/((i-1)*i)
        i += 2

    return mf*s


# функция, вычисляющая "точное" значение
def f_exact(x: float) -> float:
    return math.cosh(1+math.sqrt(1+x))*math.cos(math.sqrt(1+x-x*x))


# функция, вычисляющая приближенное значение
def f_approx(x: float) -> float:
    return cosh(1+sqrt(1+x))*cos(sqrt(1+x-x*x))


print("{:<6} {:<20} {:<20} {:<20}".format("x", "f_exact", "f_approx", "error"))
for x in interval:
    f_e = f_exact(x)
    f_a = f_approx(x)
    print("{:<6} {:<20} {:<20} {:<20}".format(x, f_e, f_a, abs(f_e-f_a)))



