import numpy as np


class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds.min, bounds.max, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = np.inf
        self.fitness = np.inf

    def update_best(self, cost_func):
        self.fitness = cost_func(self.position)
        if self.fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds.min, bounds.max)
