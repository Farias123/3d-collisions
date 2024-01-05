import numpy as np
from vpython import rate, sphere, vector, color, curve, scene

fps = 60
dt = 1/fps

x0 = 0
y0 = 0
z0 = 0
vx0 = 5
vy0 = 5
vz0 = 5
r = np.array([x0, y0, z0, vx0, vy0, vz0])

scene.camera.pos = vector(15, 5, 15)
scene.camera.axis = vector(-15, -5, -15)
outline = sphere(pos=vector(0, 0, 0), radius=10, color=color.gray(0.5), opacity=0.2)
particle_1 = sphere(pos=vector(x0, y0, z0), radius=1, color=color.red)
x = curve(vector(0, 0, 0), vector(10, 0, 0), color=color.blue)
y = curve(vector(0, 0, 0), vector(0, 10, 0), color=color.yellow)
z = curve(vector(0, 0, 0), vector(0, 0, 10), color=color.red)


def f_r(ri):
    x, y, z, vx, vy, vz = ri
    f_x = vx
    f_y = vy
    f_z = vz
    # todo
    f_vx = 0
    f_vy = -9.8
    f_vz = 0
    # todo: fazer colisões
    return np.array([f_x, f_y, f_z, f_vx, f_vy, f_vz])


def rk_4(f_r, r):
    k1 = dt * f_r(r)
    k2 = dt * f_r(r + k1 / 2)
    k3 = dt * f_r(r + k2 / 2)
    k4 = dt * f_r(r + k3)

    r = r + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return r


while True:
    rate(fps)
    r = rk_4(f_r, r)
    particle_1.pos = vector(r[0], r[1], r[2])
