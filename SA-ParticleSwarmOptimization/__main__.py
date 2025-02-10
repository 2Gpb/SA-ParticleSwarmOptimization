from types import SimpleNamespace
import numpy as np


def initialize_swarm(num_particles, dim, bounds):
    swarm = np.random.uniform(bounds.min, bounds.max, (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    return swarm, velocities


def evaluate_swarm(cost_func, swarm):
    return np.array([cost_func(particle) for particle in swarm])


def update_velocities(velocities, swarm, best_positions, swarm_best_position, w, c1, c2):
    r1 = np.random.uniform(0, 1, swarm.shape)
    r2 = np.random.uniform(0, 1, swarm.shape)
    return (
            w * velocities +
            c1 * r1 * (best_positions - swarm) +
            c2 * r2 * (swarm_best_position - swarm)
    )


def update_positions(swarm, velocities, bounds):
    swarm += velocities
    return np.clip(swarm, bounds.min, bounds.max)


def update_best_positions(swarm, fitness_values, best_positions, best_fitness):
    improved = fitness_values < best_fitness
    best_positions[improved] = swarm[improved]
    best_fitness[improved] = fitness_values[improved]
    return best_positions, best_fitness



def update_global_best(swarm, fitness_values, swarm_best_position, swarm_best_fitness):
    min_idx = np.argmin(fitness_values)
    if fitness_values[min_idx] < swarm_best_fitness:
        return swarm[min_idx].copy(), fitness_values[min_idx]
    return swarm_best_position, swarm_best_fitness


def pso(cost_func, dim=2, num_particles=30, max_iter=100, w=0.9, c1=1.5, c2=1.5, min_bound=-5.12, max_bound=5.12):
    bounds = SimpleNamespace(min=min_bound, max=max_bound)
    swarm, velocities = initialize_swarm(num_particles, dim, bounds)

    best_positions = swarm.copy()
    best_fitness = evaluate_swarm(cost_func, best_positions)

    swarm_best_position = best_positions[np.argmin(best_fitness)].copy()
    swarm_best_fitness = np.min(best_fitness)

    for i in range(max_iter):
        velocities = update_velocities(velocities, swarm, best_positions, swarm_best_position, w, c1, c2)
        swarm = update_positions(swarm, velocities, bounds)

        fitness_values = evaluate_swarm(cost_func, swarm)

        best_positions, best_fitness = update_best_positions(swarm, fitness_values, best_positions, best_fitness)

        swarm_best_position, swarm_best_fitness = update_global_best(swarm, fitness_values, swarm_best_position,
                                                                     swarm_best_fitness)

    return swarm_best_position, swarm_best_fitness


def F1(sol):
    return np.sum(np.array(sol)**2)


dimension = 10
nums_particles = 30
max_iteration = 600
w = 0.9
c1 = 1.5
c2 = 1.5
min_bound = -5.12
max_bound = 5.12


solution, fitness = pso(F1, dim=dimension, num_particles=nums_particles, max_iter=max_iteration, w=w, c1=c1, c2=c2,
                        min_bound=min_bound, max_bound=max_bound)

print("Best solution:", solution)
print("Fitness of the best solution:", fitness)
