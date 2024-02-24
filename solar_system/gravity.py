import csv
import numpy as np
from numba import jit
from vpython import rate, sphere, vector, color, scene
from general_funcs import magnitude, pos, rk4

fps = 60  # frames per second
dps = 45  # days per second of simulation
dt = dps * 86400 * 1 / fps
G = 6.67430e-11  # N*m**2/kg**2

scene.background = color.black
scene.camera.pos = vector(-9.62229e11, 1.79092e12, -1.7547e12)
scene.camera.axis = -scene.camera.pos


def general_setup(remove_bodies):
    with open('celestial_bodies_info.csv') as my_file:
        table = csv.reader(my_file)
        content = [x for x in table][1:]

        number_bodies = len(content)
        vpython_bodies = []
        structured_bodies = []
        dtype = np.dtype([('name', 'U10'), ('radius', 'f8'), ('positions', object), ('mass', 'f8'), ('r', object)])
        for i in range(number_bodies):
            b_name, b_mass, b_radius, b_speed, b_sun_distance, rgbstring = content[i]
            if b_name in remove_bodies:
                continue

            red, green, blue = rgbstring.split("/ ")
            b_rgb_vector = vector(float(red), float(green), float(blue))
            vpython_bodies.append(sphere(name=b_name, pos=vector(float(b_sun_distance), 0, 0), make_trail=True,
                                         radius=float(b_radius), color=b_rgb_vector, positions=[], mass=float(b_mass),
                                         r=np.array([float(b_sun_distance), 0, 0, 0, 0, float(b_speed)]), retain=60))
            structured_bodies.append((b_name, b_radius, [], b_mass,
                                      np.array([float(b_sun_distance), 0, 0, 0, 0, float(b_speed)])))
        dtype_bodies = np.array(structured_bodies, dtype)
        return vpython_bodies, dtype_bodies


remove = [
    # 'Sun', 'Mercury', 'Venus',
    # 'Earth', 'Mars', 'Jupiter',
    # 'Saturn', 'Uranus', 'Neptune'
]
celestial_bodies, np_bodies = general_setup(remove)


def f_r(ri, **kwargs):
    i = kwargs.get('index')
    x, y, z, vx, vy, vz = ri
    f_x, f_y, f_z = vx, vy, vz
    f_vx = 0
    f_vy = 0
    f_vz = 0

    number_bodies = len(celestial_bodies)
    for j in range(number_bodies):
        # todo aplicar dtype bodies
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

        body.pos = vector(body.r[0], body.r[1], body.r[2])
