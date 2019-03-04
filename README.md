# CS7641 Assignment 2 - Randomized Optimization

All code is located at
(github.com/astex/cs7641a2)[https://github.com/astex/cs7641a2]. I ran
everything in python 3.7.2. It all might work in some other version or it might
not.

You will first need to install mlrose, pygal, and everything pygal needs to
render PNGs. I've included a `requirements.txt` file that enumerates these
dependencies. So,

```bash
$ pip install -r requirements.txt
```

There are two main modules, `cs7641a2.main` and `cs7641a2.demos`. The first
contains code to run optimization on the neural net. The second contains the
demo problems from the paper. You can run these as follows:

```bash
$ python -m cs7641a2.main [plot_name]
$ python -m cs7641a2.demos [plot_name]
```

These commands will put a plot (or series of plots) that use the name as a
prefix in `paper/plots`. I've included the latest generation I created while
writing the paper for reference. Run the above commands with the `-h` flag for
a list of available plots.
