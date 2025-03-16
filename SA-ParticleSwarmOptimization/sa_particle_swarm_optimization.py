from types import SimpleNamespace
from particle import *
import numpy as np
import copy


class SAParticleSwarmOptimization:
    def __init__(
            self,
            fitness_function,
            dim=10,
            num_particles=30,
            iterations=100,
            w_max=0.9,
            w_min=0.4,
            c1=1.5,
            c2=0.5,
            min_bound=-5.12,
            max_bound=5.12,
            t=0.02
    ):
        self.dim = dim
        self.num_particles = num_particles
        self.iterations = iterations
        self.w_max = w_max
        self.w_min = w_min
        self.swarm = [
            Particle(fitness_function, self.dim, SimpleNamespace(min=min_bound, max=max_bound), c1, c2, t)
            for _ in range(self.num_particles)
        ]

    def __update_inertia_weight(self, best_particle):
        w = ((3 - np.exp(-self.num_particles / 200) +
              (best_particle.best_fitness / (8 * self.dim)) ** 2) ** -1 + 0.8)
        return np.clip(w, self.w_min, self.w_max)

    def optimize(self):
        best_particle = min(self.swarm, key=lambda p: p.best_fitness)

        for iter_num in range(self.iterations):
            w = self.__update_inertia_weight(best_particle)
            for particle in self.swarm:
                particle.update_velocity(w, best_particle.best_position)
                particle.update_position()
                if particle.best_fitness < best_particle.fitness:
                    best_particle = copy.deepcopy(particle)

        return best_particle.best_position, best_particle.best_fitness
