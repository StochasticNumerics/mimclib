#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import numpy as np

if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()
    from mimclib.plot import run_plot_program
    run_plot_program()
