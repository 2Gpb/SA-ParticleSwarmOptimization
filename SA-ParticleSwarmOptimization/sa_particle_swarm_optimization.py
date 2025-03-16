from types import SimpleNamespace
from particle import *
import numpy as np
import copy


class ParticleSwarmOptimization:
    def __init__(self, fitness_func,
                 dim=2,
                 num_particles=30,
                 max_iter=100,
                 w_max=0.9,
                 w_min=0.4,
                 c1=1.5,
                 c2=1.5,
                 min_bound=-5.12,
                 max_bound=5.12):
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.swarm = [Particle(fitness_func, self.dim, SimpleNamespace(min=min_bound, max=max_bound), c1, c2)
                      for _ in range(self.num_particles)]

    def __update_inertia_weight(self, best_particle):
        w = ((3 - np.exp(-self.num_particles / 200) + (best_particle.best_fitness / (8 * self.dim)) ** 2)
             ** -1 + 0.8)
        return np.clip(w, self.w_min, self.w_max)

    def optimize(self):
        best_particle = copy.deepcopy(min(self.swarm, key=lambda p: p.fitness))

        for iter_num in range(self.max_iter):
            w = self.__update_inertia_weight(best_particle)
            for particle in self.swarm:
                particle.update_velocity(w, best_particle.best_position)
                particle.update_position()
                if particle.best_fitness < best_particle.fitness:
                    best_particle = copy.deepcopy(particle)

        return best_particle.best_position, best_particle.best_fitness
