[collisions_simulation.webm](https://github.com/Farias123/3d-simulations/assets/115378746/6504bda7-8be3-496c-af09-f3f75a3694a0)
## sphere_collisions.py


[gravity_simulation.webm](https://github.com/Farias123/3d-simulations/assets/115378746/d995d3db-2142-45e8-bbfe-d96519e2b41b)
## gravity.py

-- PT-BR -- 

# 3d-simulations

Este projeto consiste em duas simulações diferentes (até agora):

## sphere_collisions.py 
Este arquivo mostra uma interação dinâmica de um grupo de partículas entre si e a fronteira de uma esfera maior que as envolve. As partículas sofrem ação da força gravitacional (da Terra) e colidem entre si, assim como com a esfera grande.

## gravity.py 
Este arquivo, por outro lado, simula o movimento de translação dos planetas do sistema solar em torno do Sol, devido à força gravitacional. A cada segundo são passados 45 dias na simulação. Todos os corpos celestiais possuem uma trilha que marca por onde passaram momentaneamente para facilitar a visualização.
Obs.: Não há simulação para colisão entre planetas.

As simulações são feitas usando o método Runge-Kutta (RK4) para resolver EDOs, e os resultados são mostrados usando a biblioteca VPython (Visual Python). Otimização da velocidade feita com numba. Ambas as simulações animadas com 60 quadros por segundo.

-- EN --

# 3d-simulations

This project consists of two different simulations (so far):

## sphere_collisions.py 
This file shows a dinamic interaction of a bunch of particles between themselves and the border of a larger sphere that involves them. The particles are affected by the gravitational force (of the Earth) and collide with each other, as well as with the large sphere.

## gravity.py 
This file on the other hand simulates the Earth's translation movement around the Sun, due to gravitational force. Every second, 45 days pass in the simulation. All the celestial bodies have trails that marks where they have passed momentarily to make it easier to visualize.
Note: The colision between planets is not simulated.

The simulations are made using the Runge-Kutta method (RK4) to solve ODEs, and the results are shown using the library VPython (Visual Python). Speed optimization with numba. Both simulations animated with 60 frames per second.
