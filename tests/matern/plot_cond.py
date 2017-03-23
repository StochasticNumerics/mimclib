#!/usr/bin/python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import mimclib.plot as miplot
import mimclib.miproj as miproj
import mimclib.db as mimcdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mimclib import ipdb
from mimclib import mimc
ipdb.set_excepthook()

mpl.rc('text', usetex=True)
mpl.rc('font', **{'family': 'normal', 'weight': 'demibold',
                  'size': 15})
rc_params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 14,
    "font.size": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    'font.weight' : 'demibold',
    'legend.fontsize' : 14}
mpl.rcParams.update(rc_params)

def plotAll(o, tags=None, label=None, db=None, work_bins=50):
    if tags is None:
        tags = [o]
    if label is None:
        label = tags

    figures = dict()
    def add_fig(name):
        if name not in figures:
            figures[name] = plt.figure()
        return figures[name].gca()

    color = ['b', 'r', 'g', 'm']
    marker = ['*', '+', 'o', 'x']
    linestyles = ['--', '-.', '-', ':', '-']

    for i, tag in enumerate(tags):
        print("Plotting", tag)
        runs = filter(lambda r: len(r.iters) > 0, db.readRuns(tag=tag,
                                                              done_flag=None))
        if len(runs) == 0:
            continue

        exact_sol, _ = miplot.estimate_exact(runs)
        fnExactErr = lambda itrs, e=exact_sol: \
                     runs[0].fn.Norm([v.calcEg() + e*-1 for v in itrs])
        miplot.set_exact_errors(runs, fnExactErr, miplot.filteritr_all)

        for r in runs:
            prev = 0
            prev_count = 0
            for itr in r.iters:
                itr.total_samples = prev_count + np.sum([
                    np.sum(miproj.default_samples_count(
                        miproj.default_basis_from_level(beta)))
                    for beta in itr.lvls_itr(prev)])
                prev_count = itr.total_samples
                prev = itr.lvls_count

        def fnItrStats(run, i):
            itr = run.iters[i]
            return [i] + [itr.db_data.user_data[0]]*3 +\
                [itr.db_data.user_data[1]]*3 + [itr.exact_error]

        xy_binned = miplot.computeIterationStats(runs, work_bins=len(runs[0].iters),
                                                 filteritr=miplot.filteritr_all,
                                                 fnItrStats=fnItrStats,
                                                 arr_fnAgg=[np.mean,
                                                            np.mean,
                                                            np.min,
                                                            np.max,
                                                            np.mean,
                                                            np.min,
                                                            np.max,
                                                            np.max])
        ax = add_fig('condo')
        #ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Matrix Size')
        ax.errorbar(xy_binned[:, 0], xy_binned[:, 1],
                    yerr=[xy_binned[:, 2], xy_binned[:, 3]],
                    color=color[i], marker=marker[i], ls='-')

        ax = add_fig('matrix')
        #ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Condition number')
        ax.errorbar(xy_binned[:, 0], xy_binned[:, 4],
                    yerr=[xy_binned[:, 5], xy_binned[:, 6]],
                    color=color[i], marker=marker[i], ls='-')
        if len(runs) != 1 or runs[0].params.min_dim != 0:
            continue;
        ax = add_fig('alpha')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of terms')
        ax.set_ylabel('Error')
        ax.plot(xy_binned[:, 0], xy_binned[:, -1], color=color[i],
                marker=marker[i], ls='-')
        C = np.polyfit(np.log(xy_binned[:, 0]), np.log(xy_binned[:, -1]), 1)
        func = lambda x: np.exp(C[1])*x**C[0]
        ax.add_line(miplot.FunctionLine2D(fn=func, linestyle='--', c='k',
                                          label='{:.3g}'.format(C[0])))

    fileName = 'output/{}.pdf'.format(o)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(fileName) as pdf:
        for key in figures.keys():
            fig = figures[key]
            for ax in fig.axes:
                miplot.__add_legend(ax)
            pdf.savefig(fig)
            plt.close(fig)

import argparse
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-db_name", type=str, action="store",
                    help="Database Name")
parser.add_argument("-db_host", type=str, action="store",
                    help="Database Host")
parser.add_argument("-db_tag", type=str, action="store",
                    help="Database Tags")
args, _ = parser.parse_known_args()

db = mimcdb.MIMCDatabase(db=args.db_name, host=args.db_host)
plotAll(o='cond_'+args.db_tag, tags=[args.db_tag], db=db)
