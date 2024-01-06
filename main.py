import numpy as np
from vpython import rate, sphere, vector, color, curve, scene

g = 9.81
fps = 60
dt = 1/fps

x0 = 0
y0 = 0
z0 = 0
vx0 = 15
vy0 = 10
vz0 = 10

scene.camera.pos = vector(15, 5, 15)
scene.camera.axis = vector(-15, -5, -15)
base_radius = 1
outline = sphere(pos=vector(0, 0, 0), radius=10, color=color.gray(0.5), opacity=0.2)
particle_1 = sphere(pos=vector(x0, y0, z0), radius=base_radius, color=color.red, r=np.array([x0, y0, z0, vx0, vy0, vz0]),
                    positions=[], make_trail=True, retain=150)
particles = [particle_1]
x = curve(vector(0, 0, 0), vector(10, 0, 0), color=color.blue)
y = curve(vector(0, 0, 0), vector(0, 10, 0), color=color.yellow)
z = curve(vector(0, 0, 0), vector(0, 0, 10), color=color.red)


def on_limit(x, y, z):
    particle_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z)) + base_radius

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


def f_r(ri):
    x, y, z, vx, vy, vz = ri
    f_x, f_y, f_z = vx, vy, vz
    f_vx = 0
    f_vy = -g
    f_vz = 0

    return np.array([f_x, f_y, f_z, f_vx, f_vy, f_vz])


while True:
    rate(fps)
    for particle in particles:
        r = particle.r
        particle.positions.append(r)
        k1 = dt * f_r(r)
        k2 = dt * f_r(r + k1 / 2)
        k3 = dt * f_r(r + k2 / 2)
        k4 = dt * f_r(r + k3)

        particle.r = r + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x, y, z, vx, vy, vz = particle.r

        if on_limit(x, y, z):
            adjust_particle(particle)
            particle_distance = particle.pos.mag
            position = vector(x, y, z)
            normal_vector = position / particle_distance
            v = vector(vx, vy, vz)
            projection = v.proj(normal_vector)
            v -= 2 * projection
            particle.r[3:6] = v.x, v.y, v.z

        particle.pos = vector(particle.r[0], particle.r[1], particle.r[2])
