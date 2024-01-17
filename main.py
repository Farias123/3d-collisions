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
e = 1

scene.background = color.white
scene.camera.pos = vector(15, 12, 15)
scene.camera.axis = vector(-15, -12, -15)
base_radius = 1
outline = sphere(pos=vector(0, 0, 0), radius=10, color=color.gray(0.5), opacity=0.2)
particle_1 = sphere(pos=vector(x0, y0, z0), radius=base_radius, color=color.red, r=np.array([x0, y0, z0, vx0, vy0, vz0]),
                    positions=[], mass = 1, make_trail=True, retain=60)
particle_2 = sphere(pos=vector(-x0, y0, z0), radius=base_radius, color=color.red, r=np.array([-x0, y0, z0, -vx0, vy0, vz0]),
                    positions=[], mass = 1, make_trail=True, retain=60)
particle_3 = sphere(pos=vector(x0, -y0, z0), radius=base_radius, color=color.red, r=np.array([x0, -y0, z0, vx0, -vy0, vz0]),
                    positions=[], mass = 1, make_trail=True, retain=60)
particle_4 = sphere(pos=vector(x0, y0, -z0), radius=base_radius, color=color.red, r=np.array([x0, y0, -z0, vx0, vy0, -vz0]),
                    positions=[], mass = 1, make_trail=True, retain=60)

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
    particle.r[:3] = x, y, z


def adjust_collision(p1, p2):
    x1, y1, z1 = p1.r[:3]
    x2, y2, z2 = p2.r[:3]

    distance_particles = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1) + np.square(z2 - z1))
    i = -1

    while distance_particles <= p1.radius + p2.radius:
        i -= 1
        p1_old_r = p1.positions[i]
        p2_old_r = p2.positions[i]
        x1, y1, z1 = p1_old_r[:3]
        x2, y2, z2 = p2_old_r[:3]
        distance_particles = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1) + np.square(z2 - z1))

    p1.pos = vector(x1, y1, z1)
    p2.pos = vector(x2, y2, z2)
    p1.r[:3] = x1, y1, z1
    p2.r[:3] = x2, y2, z2


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
    for particle1 in particles:
        r = particle1.r
        particle1.positions.append(r)

        if len(particle1.positions) > 300:
            particle1.positions.pop(0)

        particle1.r = rk4(f_r, r)
        x, y, z, vx, vy, vz = particle1.r

        if on_limit(x, y, z, particle1):
            adjust_particle(particle1)
            particle_distance = particle1.pos.mag
            position = vector(x, y, z)
            normal_vector = -position / particle_distance
            v = vector(vx, vy, vz)
            projection = v.proj(normal_vector)
            v -= 2 * projection
            particle1.r[3:6] = v.x, v.y, v.z

        for particle2 in particles:
            if particle2 is not particle1:
                distance_vector = particle2.pos - particle1.pos
                distance = distance_vector.mag
                if distance <= particle1.radius + particle2.radius:
                    adjust_collision(particle1, particle2)
                    m1, m2 = particle1.mass, particle2.mass
                    v1 = vector(particle1.r[3], particle1.r[4], particle1.r[5])
                    v2 = vector(particle2.r[3], particle2.r[4], particle2.r[5])
                    v1_projection = v1.proj(distance_vector)
                    v2_projection = v2.proj(distance_vector)
                    v1_after = m1*v1_projection + m2*(v2_projection - e*(v1_projection - v2_projection))
                    v1_after = v1_after/(m1 + m2)
                    v2_after = v1_after + e*(v1_projection - v2_projection)
                    particle1.r[3:6] = v1_after.x, v1_after.y, v1_after.z
                    particle2.r[3:6] = v2_after.x, v2_after.y, v2_after.z


        particle1.pos = vector(particle1.r[0], particle1.r[1], particle1.r[2])
