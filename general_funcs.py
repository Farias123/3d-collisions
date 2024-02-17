import numpy as np
from numba import njit, jit


# @njit(fastmath=True)
def rk4(f_r, r, dt, **kwargs):
    k1 = dt * f_r(r, **kwargs)
    k2 = dt * f_r(r + k1 / 2, **kwargs)
    k3 = dt * f_r(r + k2 / 2, **kwargs)
    k4 = dt * f_r(r + k3, **kwargs)

    r = r + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return r


def pos(particle):
    return particle.r[:3]


def vel(particle):
    return particle.r[3:6]


def magnitude(vector):
    total = 0
    for item in vector:
        total += np.square(item)

    res = np.sqrt(total)
    return res


def projection(r1, r2):
    # r1 projected on r2 axis
    dot_prod = np.dot(r1, r2)
    squared_mag = np.square(magnitude(r2))

    return dot_prod/squared_mag*r2
