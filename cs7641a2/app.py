"""A library for running main."""

import argparse
import inspect
import logging
import sys


LOGFMT = "[%(thread)d %(asctime)s] %(filename)s %(lineno)d: %(message)s"


def log_to_stdout():
    """Make logging print to stdout."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOGFMT))
    logger.addHandler(handler)


def run(main, register=None):
    """Runs an application.

    Args:
        main: The main function to execute. Should take an args object set up
            according to register as its argument.
        register: A function to register main with the argument parser. Should
            take the argument parser as its argument.
    """
    parser = argparse.ArgumentParser(description=inspect.getdoc(main))
    parser.add_argument(
            "--logtostdout",
            help="print logging to stdout.",
            action="store_true")
    parser.set_defaults(logtostdout=True)

    if register:
        register(parser)

    args = parser.parse_args()
    if args.logtostdout:
        log_to_stdout()

    main(args)
