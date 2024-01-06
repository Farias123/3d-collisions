import numpy as np
from vpython import rate, sphere, vector, color, curve, scene

g = 9.81
fps = 60
dt = 1/fps

x0 = 0
y0 = 0
z0 = 0
vx0 = 10
vy0 = 10
vz0 = 0

# scene.camera.pos = vector(15, 5, 15)
# scene.camera.axis = vector(-15, -5, -15)
outline = sphere(pos=vector(0, 0, 0), radius=10, color=color.gray(0.5), opacity=0.2)
particle_1 = sphere(pos=vector(x0, y0, z0), radius=1, color=color.red, r=np.array([x0, y0, z0, vx0, vy0, vz0]))
particles = [particle_1]
x = curve(vector(0, 0, 0), vector(10, 0, 0), color=color.blue)
y = curve(vector(0, 0, 0), vector(0, 10, 0), color=color.yellow)
z = curve(vector(0, 0, 0), vector(0, 0, 10), color=color.red)


def on_limit(x, y, z, particle):
    # x += particle.radius
    # y += particle.radius
    # z += particle.radius
    particle_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z)) + np.sqrt(3)*particle.radius

    if particle_distance >= outline.radius:
        return True
    return False


def adjust_particle(particle):

    while particle.pos.mag + np.sqrt(3)*particle.radius >= outline.radius:
        particle.pos.x *= 0.9
        particle.pos.y *= 0.9
        particle.pos.z *= 0.9

    particle.r[0] = particle.pos.x
    particle.r[1] = particle.pos.y
    particle.r[2] = particle.pos.z
    return particle.r[:3]


def f_r(ri, particle):
    x, y, z, vx, vy, vz = ri
    f_vx = 0
    f_vy = -g
    f_vz = 0

    if on_limit(x, y, z, particle):
    #     theta = np.arctan(y/x)
    #     alpha = np.arctan(vy/vx)
    #     alpha_ = theta + np.pi - alpha
    #     v = np.sqrt(np.square(vx) + np.square(vy) + np.square(vz))
    #     f_x = v*np.cos(alpha_)
    #     f_y = v*np.sin(alpha_)
    # else:
    #     f_x = vx
    #     f_y = vy
    #     f_z = vz
        x, y, z = adjust_particle(particle)
        particle_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        normal_vector = particle.pos/particle_distance
        v = vector(vx, vy, vz)
        projection = v.proj(normal_vector)
        v -= 2*projection
        f_x, f_y, f_z = v.x, v.y, v.z
    else:
        f_x, f_y, f_z = vx, vy, vz
    # todo: fazer colisões
    return np.array([f_x, f_y, f_z, f_vx, f_vy, f_vz])


while True:
    rate(fps)
    for particle in particles:
        r = particle.r
        k1 = dt * f_r(r, particle)
        k2 = dt * f_r(r + k1 / 2, particle)
        k3 = dt * f_r(r + k2 / 2, particle)
        k4 = dt * f_r(r + k3, particle)

        particle.r = r + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        particle.pos = vector(particle.r[0], particle.r[1], particle.r[2])
