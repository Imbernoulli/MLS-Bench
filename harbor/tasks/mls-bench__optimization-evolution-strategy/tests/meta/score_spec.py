"""Score spec for optimization-evolution-strategy."""
from mlsbench.scoring.dsl import *

# best_fitness: LOWER is better (minimization benchmarks; optimal = 0)
# convergence_gen: informational timing — dropped
term("best_fitness_rastrigin_30d",
    col("best_fitness_rastrigin-30d").lower().id()
    .bounded_power(bound=0.0))

term("best_fitness_rosenbrock_30d",
    col("best_fitness_rosenbrock-30d").lower().id()
    .bounded_power(bound=0.0))

term("best_fitness_ackley_30d",
    col("best_fitness_ackley-30d").lower().id()
    .bounded_power(bound=0.0))

term("best_fitness_rastrigin_100d",
    col("best_fitness_rastrigin-100d").lower().id()
    .bounded_power(bound=0.0))

setting("rastrigin-30d", weighted_mean(("best_fitness_rastrigin_30d", 1.0)))
setting("rosenbrock-30d", weighted_mean(("best_fitness_rosenbrock_30d", 1.0)))
setting("ackley-30d", weighted_mean(("best_fitness_ackley_30d", 1.0)))
setting("rastrigin-100d", weighted_mean(("best_fitness_rastrigin_100d", 1.0)))

task(gmean("rastrigin-30d", "rosenbrock-30d", "ackley-30d", "rastrigin-100d"))
