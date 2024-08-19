import numpy as np
from vpython import rate, sphere, vector, color, scene, arrow
from numba import njit
from general_funcs import rk4, pos, vel, magnitude, projection

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

dtype = np.dtype([('name', 'U10'), ('radius', 'f8'), ('last_position', object), ('mass', 'f8'), ('r', object)])

particle_1 = sphere(name='particle1', pos=vector(x0, y0, z0), radius=base_radius, color=color.red, r=np.array([x0, y0, z0, vx0, vy0, vz0]),
                    mass=5, make_trail=True, retain=60)
particle_2 = sphere(name='particle2', pos=vector(-x0, y0, z0), radius=base_radius, color=color.red, r=np.array([-x0, y0, z0, -vx0, vy0, vz0]),
                    mass=1, make_trail=True, retain=60)
particle_3 = sphere(name='particle3', pos=vector(x0, -y0, z0), radius=base_radius, color=color.red, r=np.array([x0, -y0, z0, vx0, -vy0, vz0]),
                    mass=1, make_trail=True, retain=60)
particle_4 = sphere(name='particle4', pos=vector(x0, y0, -z0), radius=base_radius, color=color.red, r=np.array([x0, y0, -z0, vx0, vy0, -vz0]),
                    mass=1, make_trail=True, retain=60)

vpython_particles = [particle_1, particle_2, particle_3, particle_4]
np_particles = []
for vp in vpython_particles:
    data = (vp.name, vp.radius, [], vp.mass, vp.r)
    np_particles.append(data)

particles = np.array(np_particles, dtype)
x = arrow(pos=vector(0, 0, 0), axis=vector(10, 0, 0), color=color.blue, shaftwidth=0.2)
y = arrow(pos=vector(0, 0, 0), axis=vector(0, 10, 0), color=color.yellow, shaftwidth=0.2)
z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 10), color=color.red, shaftwidth=0.2)


def on_limit(x, y, z, particle):
    particle_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z)) + particle['radius']

    if particle_distance >= outline.radius:
        return True
    return False


def adjust_collision(particle):
    particle_old_r = particle['last_position']
    x, y, z = particle_old_r[:3]
    particle['r'][:3] = x, y, z


@njit(fastmath=True)
def f_r(ri):
    x, y, z, vx, vy, vz = ri
    f_x, f_y, f_z = vx, vy, vz
    f_vx = 0
    f_vy = -g
    f_vz = 0

    return np.array([f_x, f_y, f_z, f_vx, f_vy, f_vz])


while True:
    rate(fps)
    for particle1 in particles:
        r = particle1['r']
        particle1['last_position'] = r

        particle1['r'] = rk4(f_r, r, dt)
        x, y, z, vx, vy, vz = particle1['r']

        for particle2 in particles:
            # sphere on sphere collision
            if particle1['name'] != particle2['name']:
                distance_vector = pos(particle2) - pos(particle1)
                distance = magnitude(distance_vector)
                if distance <= particle1['radius'] + particle2['radius']:
                    adjust_collision(particle1)
                    adjust_collision(particle2)
                    distance_vector = pos(particle2) - pos(particle1)
                    m1, m2 = particle1['mass'], particle2['mass']
                    v1 = vel(particle1)
                    v2 = vel(particle2)
                    v1_projected = projection(v1, distance_vector)
                    v2_projected = projection(v2, distance_vector)
                    v1_after = m1*v1_projected + m2*(v2_projected - e*(v1_projected - v2_projected))
                    v1_after = v1_after/(m1 + m2)
                    v2_after = v1_after + e*(v1_projected - v2_projected)
                    particle1['r'][3:6] = v1_after
                    particle2['r'][3:6] = v2_after

        if on_limit(x, y, z, particle1):
            # border collision
            adjust_collision(particle1)
            position = pos(particle1)
            scalar_distance = magnitude(position)
            normal_vector = -position / scalar_distance
            v = np.array([vx, vy, vz])
            proj = projection(v, normal_vector)
            v -= 2 * proj
            particle1['r'][3:6] = v

        vpython_p = [x for x in vpython_particles if x.name == particle1['name']][0]
        vpython_p.pos = vector(particle1['r'][0], particle1['r'][1], particle1['r'][2])
