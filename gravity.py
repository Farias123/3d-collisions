import numpy as np
from vpython import rate, sphere, vector, color, scene, arrow
from numba import njit
from general_funcs import rk4, pos, vel, magnitude, projection

fps = 60  # frames per second
dps = 30  # days per second of simulation
dt = dps*86400*1/fps
G = 6.67430e-11  # N*m**2/kg**2

r_sun = 695_700e4  # m
m_sun = 1_988_500e24  # kg
d_earth = 149_600e6  # m
r_earth = 6371e3  # m
m_earth = 5.9722e24  # kg
v_earth = 29780  # m/s

scene.background = color.black
scene.camera.pos = vector(0, 3e+10, 2.79218e+11)
scene.camera.axis = -scene.camera.pos
# scene.camera.axis = vector(-15, -12, -15)

Sun = sphere(pos=vector(0, 0, 0), radius=r_sun, color=color.yellow, r=np.array([0, 0, 0, 0, 0, 0]),
                    positions=[], mass=m_sun, make_trail=True, retain=60)
Earth = sphere(pos=vector(d_earth, 0, 0), radius=r_earth, color=color.blue, r=np.array([d_earth, 0, 0, 0, 0, v_earth]),
                    positions=[], mass=m_earth, make_trail=True, retain=60)


celestial_bodies = [Sun, Earth]


def f_r(ri, **kwargs):
    i = kwargs.get('index')
    x, y, z, vx, vy, vz = ri
    f_x, f_y, f_z = vx, vy, vz
    f_vx = 0
    f_vy = 0
    f_vz = 0

    number_bodies = len(celestial_bodies)
    for j in range(number_bodies):
        if i != j:
            body1 = celestial_bodies[i]
            body2 = celestial_bodies[j]
            M = body2.mass
            vector_d = pos(body2) - pos(body1)
            d = magnitude(vector_d)
            accell_scalar = M*G/d**2
            direction = vector_d/d
            b1_accell = accell_scalar*direction
            f_vx += b1_accell[0]
            f_vy += b1_accell[1]
            f_vz += b1_accell[2]

    return np.array([f_x, f_y, f_z, f_vx, f_vy, f_vz])


while True:
    rate(fps)
    for body in celestial_bodies:
        r = body.r
        body.positions.append(r)

        if len(body.positions) > 300:
            body.positions.pop(0)
        i = celestial_bodies.index(body)
        body.r = rk4(f_r, r, dt, index=i)
        x, y, z, vx, vy, vz = body.r

        body.pos = vector(body.r[0], body.r[1], body.r[2])
