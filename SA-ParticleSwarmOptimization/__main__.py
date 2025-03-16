from sa_particle_swarm_optimization import *
from benchmarks import *


pso = ParticleSwarmOptimization(
    fitness_func=f1,
    dim=10,
    num_particles=50,
    max_iter=1000,
    min_bound=-100,
    max_bound=100,
)

solution, fitness = pso.optimize()

print("Best solution:", solution)
print("Fitness of the best solution:", fitness)
