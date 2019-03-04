"""The main executable for california data."""

import logging
import pygal
import sklearn.datasets
import sklearn.preprocessing

from . import app
from . import common as lib
from . import neuralnet


def get_data(datadir):
    """Gets california housing samples and labels."""
    x, y = sklearn.datasets.fetch_california_housing(
        datadir,
        return_X_y=True)
    (lx, ly), (tx, ty) = lib.partition((x, y), (.8, .2))

    scaler = sklearn.preprocessing.StandardScaler().fit(lx)
    lx = scaler.transform(lx)
    tx = scaler.transform(tx)

    binarizer = lib.MedianBinarizer().fit(ly)
    ly = binarizer.transform(ly)
    ty = binarizer.transform(ty)

    return (lx, ly), (tx, ty)


def register(parser):
    neuralnet.register()

    parser.add_argument(
        "--datadir",
        help="cache directory for scikit data",
        default="data")
    parser.add_argument(
        "--outdir",
        help="dir for output plots",
        default="paper/plots")
    parser.add_argument(
        "plotfuncs",
        nargs="+",
        help="variables to plot against",
        choices=lib.get_plotfuncs() + ["all", "neuralnet_nodes"])


def plot_neuralnet_nodes(outdir, ldata, tdata):
    funcs = [
        lib.get_plotfunc("neuralnet_annealing_2node_max_iter"),
        lib.get_plotfunc("neuralnet_annealing_10node_max_iter"),
        lib.get_plotfunc("neuralnet_annealing_20node_max_iter")]
    titles = ["2 nodes", "10 nodes", "20 nodes"]
    plots = [lib.Plotter("max_iter") for _ in range(3)]

    for func, plot in zip(funcs, plots):
        func(ldata, tdata, plot)

    splot = pygal.XY(stroke=False, x_title="max_iter", y_title="score")
    tplot = pygal.XY(stroke=False, x_title="max_iter", y_title="fit_time")
    for plotter, title in zip(plots, titles):
        splot.add(title + " training", plotter.lerr)
        splot.add(title + " testing", plotter.terr)
        tplot.add(title, plotter.ftimes)
    splot.render_to_file("%s/neuralnet_nodes.svg" % (outdir,))
    splot.render_to_png("%s/neuralnet_nodes.png" % (outdir,))
    tplot.render_to_file("%s/neuralnet_nodes_ftime.svg" % (outdir,))
    tplot.render_to_png("%s/neuralnet_nodes_ftime.png" % (outdir,))


def main(args):
    logging.info("prepping data...")
    ldata, tdata = get_data(args.datadir)

    plotfuncs = (
        lib.get_plotfuncs() if "all" in args.plotfuncs else
        args.plotfuncs)

    if "neuralnet_nodes" in plotfuncs:
        logging.info("plotting 2/10/20 node graph...")
        plotfuncs.remove("neuralnet_nodes")
        plot_neuralnet_nodes(args.outdir, ldata, tdata)

    for plotfunc in plotfuncs:
        plot = lib.get_plotfunc(plotfunc)
        xtitle = lib.get_xtitle(plotfunc)

        logging.info("plotting %s...", plotfunc)
        plotter = lib.Plotter(xtitle)
        try:
            plot(ldata, tdata, plotter)
        except KeyboardInterrupt:
            logging.info("caught keyboard interrupt. plotting and continuing.")
            plotter.write(args.outdir, plotfunc)
            continue
        except Exception:
            logging.exception("error in %s. continuing...", plotfunc)
            continue

        plotter.write(args.outdir, plotfunc)


if __name__ == "__main__":
    app.run(main, register)
