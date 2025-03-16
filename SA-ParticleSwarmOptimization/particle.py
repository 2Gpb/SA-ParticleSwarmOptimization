import numpy as np


class Particle:
    def __init__(self, fitness_func, dim, bounds, c1, c2):
        self.fitness_func = fitness_func
        self.dim = dim
        self.bounds = bounds

        self.c1 = c1
        self.c2 = c2

        self.position = np.random.uniform(bounds.min, bounds.max, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = self.fitness = self.fitness_func(self.position)

    def update_velocity(self, w, global_best_pos):
        r1 = np.random.uniform(0, 1, self.dim)
        r2 = np.random.uniform(0, 1, self.dim)
        inertia = w * self.velocity
        cognitive = self.c1 * r1 * (self.best_position - self.position)
        social = self.c2 * r2 * (global_best_pos - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self):
        self.position = np.clip(self.position + self.velocity, self.bounds.min, self.bounds.max)
        self._update_best()

    def _update_best(self):
        self.fitness = self.fitness_func(self.position) * 0.01
        if self.fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness
