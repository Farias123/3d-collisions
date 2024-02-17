import numpy as np
from vpython import rate, sphere, vector, color, scene
from general_funcs import magnitude, pos, rk4

fps = 60  # frames per second
dps = 30  # days per second of simulation
dt = dps * 86400 * 1 / fps
G = 6.67430e-11  # N*m**2/kg**2

r_sun = 695_700e4  # m
m_sun = 1_988_500e24  # kg

d_mercury = 57_909e6  # m
r_mercury = 2439.7e3  # m
m_mercury = 0.33010e24  # kg
v_mercury = 47360  # m/s

d_venus = 108_210.5e6  # m
r_venus = 6051.8e3  # m
m_venus = 4.8673e24  # kg
v_venus = 35020  # m/s

d_earth = 149_600e6  # m
r_earth = 6371e3  # m
m_earth = 5.9722e24  # kg
v_earth = 29780  # m/s

d_mars = 227_955.5e6  # m
r_mars = 3389.5e3  # m
m_mars = 0.64169e24  # kg
v_mars = 24080  # m/s

d_jupiter = 778_479e6  # m
r_jupiter = 69911e3  # m
m_jupiter = 1898.13e24  # kg
v_jupiter = 13060  # m/s

d_saturn = 1_432_040.5e6  # m
r_saturn = 58_232e3  # m
m_saturn = 568.32e24  # kg
v_saturn = 9670  # m/s

d_uranus = 2_867_043e6  # m
r_uranus = 25_362e3  # m
m_uranus = 86.811e24  # kg
v_uranus = 6790  # m/s

d_neptune = 4514953.5e6  # m
r_neptune = 24622e3  # m
m_neptune = 102.409e24  # kg
v_neptune = 5450  # m/s

scene.background = color.black
scene.camera.pos = vector(-9.62229e11, 1.79092e12, -1.7547e12)
scene.camera.axis = -scene.camera.pos

Sun = sphere(name="Sun", pos=vector(0, 0, 0), radius=r_sun, color=color.yellow, r=np.array([0, 0, 0, 0, 0, 0]),
             positions=[], mass=m_sun, make_trail=True, retain=60)
Mercury = sphere(name="Mercury", pos=vector(d_mercury, 0, 0), radius=r_mercury, color=color.gray(0.5),
                 r=np.array([d_mercury, 0, 0, 0, 0, v_mercury]),
                 positions=[], mass=m_mercury, make_trail=True, retain=60)
Venus = sphere(name="Venus", pos=vector(d_venus, 0, 0), radius=r_venus, color=color.orange,
               r=np.array([d_venus, 0, 0, 0, 0, v_venus]),
               positions=[], mass=m_venus, make_trail=True, retain=60)
Earth = sphere(name="Earth", pos=vector(d_earth, 0, 0), radius=r_earth, color=color.blue,
               r=np.array([d_earth, 0, 0, 0, 0, v_earth]),
               positions=[], mass=m_earth, make_trail=True, retain=60)
Mars = sphere(name="Mars", pos=vector(d_mars, 0, 0), radius=r_mars, color=color.red,
              r=np.array([d_mars, 0, 0, 0, 0, v_mars]),
              positions=[], mass=m_mars, make_trail=True, retain=60)
Jupiter = sphere(name="Jupiter", pos=vector(d_jupiter, 0, 0), radius=r_jupiter, color=color.orange,
                 r=np.array([d_jupiter, 0, 0, 0, 0, v_jupiter]),
                 positions=[], mass=m_jupiter, make_trail=True, retain=60)
Saturn = sphere(name="Saturn", pos=vector(d_saturn, 0, 0), radius=r_saturn, color=color.yellow,
                r=np.array([d_saturn, 0, 0, 0, 0, v_saturn]),
                positions=[], mass=m_saturn, make_trail=True, retain=60)
Uranus = sphere(name="Uranus", pos=vector(d_uranus, 0, 0), radius=r_uranus, color=color.cyan,
                r=np.array([d_uranus, 0, 0, 0, 0, v_uranus]),
                positions=[], mass=m_uranus, make_trail=True, retain=60)
Neptune = sphere(name="Neptune", pos=vector(d_neptune, 0, 0), radius=r_neptune, color=color.blue,
                 r=np.array([d_neptune, 0, 0, 0, 0, v_neptune]),
                 positions=[], mass=m_neptune, make_trail=True, retain=60)

celestial_bodies = [Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune]


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
            accell_scalar = M * G / d ** 2
            direction = vector_d / d
            b1_accell = accell_scalar * direction
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

        print(scene.camera.pos, scene.camera.axis)
        body.pos = vector(body.r[0], body.r[1], body.r[2])
