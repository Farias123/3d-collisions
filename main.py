import numpy as np
from vpython import rate, sphere, vector, color, scene, arrow, text
from numba import njit

g = 9.81
fps = 60
dt = 1/fps

x0 = 4
y0 = 4
z0 = 6
vx0 = 4
vy0 = 2
vz0 = 1

scene.background = color.white
scene.camera.pos = vector(15, 12, 15)
scene.camera.axis = vector(-15, -12, -15)
base_radius = 1
outline = sphere(pos=vector(0, 0, 0), radius=10, color=color.gray(0.5), opacity=0.2)
particle_1 = sphere(pos=vector(x0, y0, z0), radius=base_radius, color=color.red, r=np.array([x0, y0, z0, vx0, vy0, vz0]),
                    positions=[], make_trail=True, retain=60)
particle_2 = sphere(pos=vector(x0, y0, z0), radius=base_radius, color=color.red, r=np.array([x0, y0, z0, -vx0, vy0, vz0]),
                    positions=[], make_trail=True, retain=60)
particle_3 = sphere(pos=vector(x0, y0, z0), radius=base_radius, color=color.red, r=np.array([x0, y0, z0, vx0, -vy0, vz0]),
                    positions=[], make_trail=True, retain=60)
particle_4 = sphere(pos=vector(x0, y0, z0), radius=base_radius, color=color.red, r=np.array([x0, y0, z0, vx0, vy0, -vz0]),
                    positions=[], make_trail=True, retain=60)

particles = [particle_1, particle_2, particle_3, particle_4]
x = arrow(pos=vector(0, 0, 0), axis=vector(10, 0, 0), color=color.blue, shaftwidth=0.2)
y = arrow(pos=vector(0, 0, 0), axis=vector(0, 10, 0), color=color.yellow, shaftwidth=0.2)
z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 10), color=color.red, shaftwidth=0.2)


def on_limit(x, y, z, particle):
    particle_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z)) + particle.radius

    if particle_distance >= outline.radius:
        return True
    return False


def adjust_particle(particle):
    x, y, z = particle.r[:3]
    particle_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z)) + particle.radius
    outer_radius = outline.radius
    i = -1

    while particle_distance >= outer_radius:
        i -= 1
        particle_old_r = particle.positions[i]
        x, y, z = particle_old_r[:3]
        particle_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z)) + particle.radius

    particle.pos = vector(x, y, z)


@njit(fastmath=True)
def f_r(ri):
    x, y, z, vx, vy, vz = ri
    f_x, f_y, f_z = vx, vy, vz
    f_vx = 0
    f_vy = -g
    f_vz = 0

    return np.array([f_x, f_y, f_z, f_vx, f_vy, f_vz])


@njit(fastmath=True)
def rk4(f_r, r):
    k1 = dt * f_r(r)
    k2 = dt * f_r(r + k1 / 2)
    k3 = dt * f_r(r + k2 / 2)
    k4 = dt * f_r(r + k3)

    r = r + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return r


while True:
    rate(fps)
    for particle in particles:
        r = particle.r
        particle.positions.append(r)

        if len(particle.positions) > 300:
            particle.positions.pop(0)

        particle.r = rk4(f_r, r)
        x, y, z, vx, vy, vz = particle.r

        if on_limit(x, y, z, particle):
            adjust_particle(particle)
            particle_distance = particle.pos.mag
            position = vector(x, y, z)
            normal_vector = -position / particle_distance
            v = vector(vx, vy, vz)
            projection = v.proj(normal_vector)
            v -= 2 * projection
            particle.r[3:6] = v.x, v.y, v.z

        particle.pos = vector(particle.r[0], particle.r[1], particle.r[2])
