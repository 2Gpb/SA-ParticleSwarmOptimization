import numpy as np


class Particle:
    def __init__(self, fitness_function, dim, bounds, c1, c2, t):
        self.fitness_function = fitness_function
        self.dim = dim
        self.bounds = bounds

        self.c1 = c1
        self.c2 = c2
        self.t = t

        self.position = np.random.uniform(bounds.min, bounds.max, dim)
        self.best_position = self.position.copy()
        self.fitness = self.best_fitness = self.fitness_function(self.best_position)
        self.velocity = np.random.uniform(-1, 1, dim)

    def update_velocity(self, w, global_best_position):
        r1, r2 = np.random.rand(2, self.dim)
        inertia = w * self.velocity
        local = self.c1 * r1 * (self.best_position - self.position)
        social = self.c2 * r2 * (global_best_position - self.position)
        self.velocity = inertia + local + social

    def update_position(self):
        self.position = np.clip(self.position + self.velocity, self.bounds.min, self.bounds.max)
        self.__update_best()

    def __update_best(self):
        self.fitness = self.fitness_function(self.position) * self.t
        if self.fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness
