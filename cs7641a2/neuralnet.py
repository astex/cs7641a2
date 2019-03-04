"""Plotting functions that train a neural net."""

import mlrose
import mlrose.decay
import numpy as np
from . import common as lib


ITER_MIN = 500
ITER_MAX = 10_000
ITER_STEP = 500


def iterfunc(algorithm, factor=1, nodes=None):
    def plotfunc(ldata, tdata, plotter):
        for max_iter in range(
                int(ITER_MIN / factor),
                int(ITER_MAX / factor + 1),
                int(ITER_STEP / factor)):
            classifier = mlrose.NeuralNetwork(
                nodes or [10],
                algorithm=algorithm,
                max_iters=max_iter)
            plotter.plot(classifier, max_iter, ldata, tdata)
    return plotfunc


NODE_MIN = 1
NODE_MAX = 20
NODE_STEP = 1


def nodefunc(algorithm, factor=1):
    def plotfunc(ldata, tdata, plotter):
        for nodes in range(NODE_MIN, NODE_MAX + 1, NODE_STEP):
            classifier = mlrose.NeuralNetwork(
                [nodes],
                algorithm=algorithm,
                max_iters=4000 / factor)
            plotter.plot(classifier, nodes, ldata, tdata)
    return plotfunc


DELTA_MIN = .02
DELTA_MAX = .5
DELTA_STEP = .02


def deltafunc(algorithm):
    def plotfunc(ldata, tdata, plotter):
        delta = DELTA_MIN
        while delta < DELTA_MAX + 1e-8:
            classifier = mlrose.NeuralNetwork(
                [10],
                algorithm=algorithm,
                max_iters=4000,
                learning_rate=delta)
            plotter.plot(classifier, delta, ldata, tdata)
            delta += DELTA_STEP
    return plotfunc


TEMP_MIN = .5
TEMP_MAX = 10
TEMP_STEP = .5


def tempfunc():
    def plotfunc(ldata, tdata, plotter):
        temp = TEMP_MIN
        while temp < TEMP_MAX + 1e-8:
            classifier = mlrose.NeuralNetwork(
                    [10],
                    algorithm="simulated_annealing",
                    max_iters=2000,
                    schedule=mlrose.decay.GeomDecay(temp))
            plotter.plot(classifier, temp, ldata, tdata)
            temp += TEMP_STEP
    return plotfunc


hillclimb_iter = iterfunc("random_hill_climb")
hillclimb_node = nodefunc("random_hill_climb")
annealing_iter = iterfunc("simulated_annealing")
annealing_node = nodefunc("simulated_annealing")
genetic_iter = iterfunc("genetic_alg", 50)
genetic_node = nodefunc("genetic_alg", 50)

annealing_iter_2 = iterfunc("simulated_annealing", nodes=[2])
annealing_iter_10 = iterfunc("simulated_annealing", nodes=[10])
annealing_iter_20 = iterfunc("simulated_annealing", nodes=[20])

hillclimb_delta = deltafunc("random_hill_climb")
annealing_delta = deltafunc("random_hill_climb")

annealing_temp = tempfunc()


def register():
    np.seterr(over="ignore")
    lib.register_plotfunc("neuralnet_hillclimb", "max_iter", hillclimb_iter)
    lib.register_plotfunc("neuralnet_hillclimb", "hidden_nodes", hillclimb_node)
    lib.register_plotfunc("neuralnet_annealing", "max_iter", annealing_iter)
    lib.register_plotfunc("neuralnet_annealing", "hidden_nodes", annealing_node)
    lib.register_plotfunc("neuralnet_genetic", "max_iter", genetic_iter)
    lib.register_plotfunc("neuralnet_genetic", "hidden_nodes", genetic_node)
    lib.register_plotfunc(
        "neuralnet_annealing_2node", "max_iter", annealing_iter_2)
    lib.register_plotfunc(
        "neuralnet_annealing_10node", "max_iter", annealing_iter_10)
    lib.register_plotfunc(
        "neuralnet_annealing_20node", "max_iter", annealing_iter_20)
    lib.register_plotfunc("neuralnet_hillclimb", "delta", hillclimb_delta)
    lib.register_plotfunc("neuralnet_annealing", "delta", annealing_delta)
    lib.register_plotfunc("neuralnet_annealing", "init_temp", annealing_temp)
