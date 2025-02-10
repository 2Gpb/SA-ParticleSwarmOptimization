from types import SimpleNamespace
from particle import *


class ParticleSwarmOptimization:
    def __init__(self, cost_func, dim=2, num_particles=30, max_iter=100, w=0.9, c1=1.5, c2=1.5, min_bound=-5.12,
                 max_bound=5.12):
        self.cost_func = cost_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = SimpleNamespace(min=min_bound, max=max_bound)
        self.swarm = [Particle(self.dim, self.bounds) for _ in range(self.num_particles)]
        self.swarm_best_position = None
        self.swarm_best_fitness = np.inf

    def __update_velocities(self):
        for particle in self.swarm:
            r1 = np.random.uniform(0, 1, self.dim)
            r2 = np.random.uniform(0, 1, self.dim)
            inertia = self.w * particle.velocity
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

        best_particle = min(self.swarm, key=lambda part: particle.best_fitness)
        self.swarm_best_position = best_particle.best_position.copy()
        self.swarm_best_fitness = best_particle.best_fitness

        for _ in range(self.max_iter):
            self.__update_velocities()
            self.__update_positions()
            self.__update_best_positions()
            self.__update_global_best()

        return self.swarm_best_position, self.swarm_best_fitness


def f1(sol):
    return np.sum(np.array(sol)**2)


dimension1 = 10
num_particles1 = 50
max_iter1 = 1000
w1 = 0.9
c11 = 1.5
c21 = 1.5
min_bound1 = -5.12
max_bound1 = 5.12


pso = ParticleSwarmOptimization(
    cost_func=f1,
    dim=dimension1,
    num_particles=num_particles1,
    max_iter=max_iter1,
    w=w1,
    c1=c11,
    c2=c21,
    min_bound=min_bound1,
    max_bound=max_bound1
)

solution, fitness = pso.optimize()

print("Best solution:", solution)
print("Fitness of the best solution:", fitness)
