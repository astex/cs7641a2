"""A common module for working with data sets."""

import logging
import numpy
import pygal
import random
import sklearn.metrics
import sklearn.preprocessing
import time


def partition(datasets, probs):
    """Splits data into two partitions.
    
    Args:
        datasets: Any number of datasets of the same size. These will be
            partitioned together, so if the first element of one dataset goes
            to the right then the first element of all the other datasets will
            do the same.
        probs: The probability that the data is split into each of the buckets.
        
    Returns:
        A number of buckets containing some portion of the datasets according
        to probs.
    """
    buckets = [[] for _ in probs]
    for item in zip(*datasets):
        coin = random.uniform(0, 1)
        ptotal = 0
        for bucket_index, pcurr in enumerate(probs):
            ptotal += pcurr
            if coin < ptotal:
                break
        else:
            continue
        buckets[bucket_index].append(item)
    return [list(zip(*d)) for d in buckets]


class MedianBinarizer:
    """Bins data as: <med = 0 and >med = 1."""
    def __init__(self):
        self._binarizer = None

    def fit(self, x):
        median = numpy.median(x)
        self._binarizer = sklearn.preprocessing.Binarizer(median)\
            .fit([[i] for i in x])
        return self

    def transform(self, x):
        if self._binarizer is None:
            raise TypeError("fit has not been called")
        return self._binarizer.transform([[i] for i in x])


def make_boolean(data):
    """Transforms labels into two classes (>avg, <avg)."""
    samples, labels = data
    average = sum(labels) / len(labels)
    labels = [1 if label > average else -1 for label in labels]
    return samples, labels


PLOTFUNCS = {}


def register_plotfunc(prefix, x_title, plotfunc):
    PLOTFUNCS["%s_%s" % (prefix, x_title)] = (plotfunc, x_title)


def get_plotfuncs():
    return list(PLOTFUNCS.keys())


def get_plotfunc(plotfunc):
    return PLOTFUNCS[plotfunc][0]


def get_xtitle(plotfunc):
    return PLOTFUNCS[plotfunc][1]


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


class Plotter:
    def __init__(self, xtitle):
        self.lerr = []
        self.terr = []
        self.ftimes = []
        self.stimes = []
        self.xtitle = xtitle

    @property
    def learning_plot(self):
        plot = pygal.XY(
            stroke=False,
            x_title=self.xtitle,
            y_title="error")
        plot.add("training", self.lerr)
        plot.add("testing", self.terr)
        return plot

    @property
    def fit_timing_plot(self):
        plot = pygal.XY(
            stroke=False,
            show_legend=False,
            x_title=self.xtitle,
            y_title="fit_time")
        plot.add("", self.ftimes)
        return plot

    @property
    def score_timing_plot(self):
        plot = pygal.XY(
            stroke=False,
            show_legend=False,
            x_title=self.xtitle,
            y_title="score_time")
        plot.add("", self.stimes)
        return plot

    def plot(self, classifier, xval, ldata, tdata):
        logging.info("plotting data point %s...", xval)
        timer = _timer()
        classifier.fit(*ldata)
        self.ftimes.append((xval, next(timer)))

        lx, ly = ldata
        lscore = 1 - sklearn.metrics.accuracy_score(ly, classifier.predict(lx))
        self.lerr.append((xval, lscore))
        tx, ty = tdata
        tscore = 1 - sklearn.metrics.accuracy_score(ty, classifier.predict(tx))
        self.terr.append((xval, tscore))
        self.stimes.append((xval, next(timer)))

    def write(self, outdir, name):
        def write_plot(plot, suffix=""):
            plot.render_to_file(outdir + "/%s%s.svg" % (name, suffix))
            plot.render_to_png(outdir + "/%s%s.png" % (name, suffix))

        write_plot(self.learning_plot)
        write_plot(self.fit_timing_plot, "_ftime")
        write_plot(self.score_timing_plot, "_stime")
