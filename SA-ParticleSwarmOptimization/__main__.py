from sa_particle_swarm_optimization import *
import benchmarks
import inspect
from time import time


def test(n=50, iterations=1000):
    functions = sorted(list(
        filter(lambda x: x[0].startswith("f"), inspect.getmembers(
            benchmarks, inspect.isfunction))
    ), key=lambda x: int(x[0][1:]))

    for func in functions:
        func = func[1]
        function_name, lb, up, dim = benchmarks.get_function_param(func.__name__)
        start_time = time()
        sa_pso = SAParticleSwarmOptimization(
            fitness_function=func,
            dim=dim,
            num_particles=n,
            iterations=iterations,
            min_bound=lb,
            max_bound=up
        )

        best_solution, best_score = sa_pso.optimize()
        time_s = time() - start_time
        print(f'function_name = {function_name}\n'
              f'time_s = {time_s}\n'
              f'best_score = {best_score}\n'
              f'best_solution = {best_solution}')
        print("__________________________________________________________________________")


def main():
    test()


if __name__ == "__main__":
    main()
