from types import SimpleNamespace
from particle import *


class ParticleSwarmOptimization:
    def __init__(self, cost_func, dim=2, num_particles=30, max_iter=100, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5,
                 min_bound=-5.12, max_bound=5.12, adaptive_w=False):
        self.cost_func = cost_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.bounds = SimpleNamespace(min=min_bound, max=max_bound)
        self.swarm = [Particle(self.dim, self.bounds) for _ in range(self.num_particles)]
        self.swarm_best_position = None
        self.swarm_best_fitness = np.inf
        self.adaptive_w = adaptive_w

    def _update_inertia_weight(self, iter_num):
        if self.adaptive_w:
            size = self.num_particles
            dimension = self.dim
            fitness_value = self.swarm_best_fitness
            return (3 - np.exp(-size / 200) + (fitness_value / (8 * dimension)) ** 2) ** -1 + 0.8
        else:
            return self.w_max - ((self.w_max - self.w_min) * iter_num) / self.max_iter

    def __update_velocities(self, w):
        for particle in self.swarm:
            r1 = np.random.uniform(0, 1, self.dim)
            r2 = np.random.uniform(0, 1, self.dim)
            inertia = w * particle.velocity
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.swarm_best_position - particle.position)
            particle.velocity = inertia + cognitive + social

    def __update_positions(self):
        for particle in self.swarm:
            particle.update_position(self.bounds)

    def __update_best_positions(self):
        for particle in self.swarm:
            particle.update_best(self.cost_func)

    def __update_global_best(self):
        for particle in self.swarm:
            if particle.best_fitness < self.swarm_best_fitness:
                self.swarm_best_position = particle.best_position.copy()
                self.swarm_best_fitness = particle.best_fitness

    def optimize(self):
        for particle in self.swarm:
            particle.update_best(self.cost_func)

        best_particle = min(self.swarm, key=lambda part: part.best_fitness)
        self.swarm_best_position = best_particle.best_position.copy()
        self.swarm_best_fitness = best_particle.best_fitness

        for iter_num in range(self.max_iter):
            w = self._update_inertia_weight(iter_num)
            self.__update_velocities(w)
            self.__update_positions()
            self.__update_best_positions()
            self.__update_global_best()

        return self.swarm_best_position, self.swarm_best_fitness


def f1(sol):
    return np.sum(np.array(sol)**2)


pso = ParticleSwarmOptimization(
    cost_func=f1,
    dim=3,
    num_particles=30,
    max_iter=1000,
    min_bound=-5.12,
    max_bound=5.12,
    adaptive_w=True
)

solution, fitness = pso.optimize()

print("Best solution:", solution)
print("Fitness of the best solution:", fitness)
