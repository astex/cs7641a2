"""Some problems that demonstrate the pros/cons of various algorithms."""


import itertools
import logging
import mlrose
import numpy
import pygal
import time

from . import app


def _timer():
    def backend():
        curr = None
        prev = time.time()
        while True:
            curr = time.time()
            yield curr - prev
            prev = curr
    timer = backend()
    next(timer)
    return timer


N_MIN = 5
N_MAX = 200
N_STEP = 5


class Problem(mlrose.DiscreteOpt):
    def find_neighbors(self):
        self.neighbors = []
        for i in range(len(self.state)):
            neighbor = numpy.copy(self.state)
            neighbor[i] = 0 if neighbor[i] == 1 else 1
            self.neighbors.append(neighbor)


def getdata(algorithm, fitness):
    """Generates data for a fitness function.
    
    Generates the following data (against the number of inputs):
        
        - Number of calls to the fitness function before maximizing the value.
        - Time to maximize the fitness function.
        - The maximum fitness according to the algorithm.

    Returns:
        (n_inputs, n_calls, time, max_fitness)
    """
    logging.info(
        "plotting %s fitness with %s...",
        fitness.__name__,
        algorithm.__name__)

    n_inputs = []
    n_calls = []
    time = []
    max_fitnesses = []
    timer = _timer()

    for n in range(N_MIN, N_MAX + N_STEP, N_STEP):
        logging.info("plotting %s inputs...", n)
        n_inputs.append(n)

        call_count = itertools.count()
        def fitness_wrapper(x):
            next(call_count)
            return fitness(x)

        problem = Problem(n, mlrose.CustomFitness(fitness_wrapper))
        next(timer)
        _, max_fitness = algorithm(problem)

        n_calls.append(next(call_count) - 1)
        time.append(next(timer))
        max_fitnesses.append(max_fitness)

    return n_inputs, n_calls, time, max_fitnesses


def hillclimb(problem):
    return mlrose.random_hill_climb(problem, max_attempts=20)


def annealing(problem):
    return mlrose.simulated_annealing(problem, max_attempts=20)


def genetic(problem):
    return mlrose.genetic_alg(problem, max_attempts=20)


def mimic(problem):
    return mlrose.mimic(problem, max_attempts=20)


class Plot:
    algorithms = [
        ("hillclimb", mlrose.random_hill_climb),
        ("annealing", mlrose.simulated_annealing),
        ("genetic", genetic),
        ("mimic", mimic),
    ]

    def __init__(self, fitness):
        self.fitness = fitness
        self.data = {}

    def run(self):
        for name, algorithm in self.algorithms:
            self.data[name] = getdata(algorithm, self.fitness)

    @property
    def n_calls_plot(self):
        plot = pygal.XY(
            stroke=False,
            x_title="n_inputs",
            y_title="n_calls")
        for name, _ in self.algorithms:
            n_inputs, n_calls, _, _ = self.data[name]
            plot.add(name, list(zip(n_inputs, n_calls)))
        return plot

    @property
    def time_plot(self):
        plot = pygal.XY(
            stroke=False,
            x_title="n_inputs",
            y_title="time")
        for name, _ in self.algorithms:
            n_inputs, _, time, _ = self.data[name]
            plot.add(name, list(zip(n_inputs, time)))
        return plot

    @property
    def fitness_plot(self):
        plot = pygal.XY(
            stroke=False,
            x_title="n_inputs",
            y_title="max_fitness")
        for name, _ in self.algorithms:
            n_inputs, _, _, fitnesses = self.data[name]
            plot.add(name, list(zip(n_inputs, fitnesses)))
        return plot

    def write(self, outdir, name):
        def write_plot(plot, suffix=""):
            plot.render_to_file(outdir + "/%s%s.svg" % (name, suffix))
            plot.render_to_png(outdir + "/%s%s.png" % (name, suffix))

        write_plot(self.n_calls_plot, "_n_calls")
        write_plot(self.time_plot, "_time")
        write_plot(self.fitness_plot, "_fitness")


def split_similarity(x):
    """Splits a vector in two and evaluates how many elements are equal."""
    split = int(len(x) / 2)
    total = sum(1 for i in range(split) if x[i] == x[split + i])
    split = int(split / 2)
    if split == 0:
        return
    total -= sum(1 for i in range(split) if x[i] == x[split + i])
    total -= sum(1 for i in range(split) if x[2 * split] == x[3 * split + i])
    return max(total, 0)

def onecount(x):
    """Counts the number of ones."""
    return sum(i for i in x if i == 1)


def switchcount(x):
    """Counts the number of inversions ([1, 0] or [0, 1])."""
    return sum(1 for i in range(1, len(x)) if x[i] != x[i - 1])


def fourpeaks(x):
    """Evaluates the four peaks function on x."""
    ones, = numpy.where(x == 1)
    zeros, = numpy.where(x == 0)
    n = len(x)
    tail = (n - ones[-1] if len(ones) else n)
    head = (zeros[0] if len(zeros) else n)
    return max(head, tail) + (n if tail > n / 4 and head > n / 4 else 0)


FUNCS = {
    "split_similarity": split_similarity,
    "onecount": onecount,
    "switchcount": switchcount,
    "fourpeaks": fourpeaks,
}


def register(parser):
    parser.add_argument(
        "funcs",
        nargs="+",
        help="fitness functions to plot",
        choices=FUNCS)


def main(args):
    for func in args.funcs:
        fitness = FUNCS[func]
        plot = Plot(fitness)
        plot.run()
        plot.write("paper/plots", func)


if __name__ == "__main__":
    app.run(main, register)
