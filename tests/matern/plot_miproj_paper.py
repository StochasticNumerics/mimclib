#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import mimclib.plot as miplot
import matplotlib.pyplot as plt
from mimclib import ipdb

def plotSeeds(ax, runs, *args, **kwargs):
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Dim')
    ax.set_ylabel('Error')
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    iter_idx = kwargs.pop('iter_idx', None)
    fnNorm = kwargs.pop("fnNorm", np.abs)
    ##### TEMP
    #itr = runs[0].last_itr
    if iter_idx is None:
        itr = runs[0].last_itr
    else:
        itr = runs[0].iters[iter_idx]
    El = itr.calcDeltaEl()
    inds = []
    x = []
    for d in xrange(1, itr.lvls_max_dim()):
        ei = np.zeros(d)
        ei[-1] = 1
        # if len(ei) >= 2:
        #     ei[-2] = 1
        ii = itr.lvls_find(ei)
        if ii is not None:
            inds.append(ii)
            x.append(d)
    inds = np.array(inds)
    x = np.array(x)
    line = ax.plot(x, fnNorm(El[inds]), *args, **kwargs)

    if Ref_kwargs is not None:
        ax.add_line(miplot.FunctionLine2D.ExpLine(data=line[0].get_xydata(),
                                                  **Ref_kwargs))
    return line[0].get_xydata(), [line]

def plotBestNTerm(ax, runs, *args, **kwargs):
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel('Error')
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    iter_idx = kwargs.pop('iter_idx', None)
    ##### TEMP
    #itr = runs[0].last_itr
    if iter_idx is None:
        itr = runs[0].last_itr
    else:
        itr = runs[0].iters[iter_idx]
    sorted_coeff = np.sort(np.abs(itr.calcEg().coefficients))[::-1]

    error = np.cumsum(np.abs(sorted_coeff[::-1]))[::-1]
    N = 2 * np.arange(1, len(sorted_coeff)+1)
    N[1] = 4
    line = ax.plot(N, error, *args, **kwargs)
    if Ref_kwargs is not None:
        sel = np.zeros(len(N), dtype=np.bool)
        #sel[np.arange(int(0.01*len(N)), int(0.03*len(N)))] = True
        sel[50:500] = True
        sel = np.logical_and(sel, error > 1e-8)
        ax.add_line(miplot.FunctionLine2D.ExpLine(data=line[0].get_xydata()[sel, :],
                                                  **Ref_kwargs))
    return line[0].get_xydata(), [line]


def plotUserData(ax, runs, *args, **kwargs):
    which = kwargs.pop('which', 'cond').lower()
    def fnItrStats(run, i):
        itr = run.iters[i]
        max_cond = np.max([d.max_cond for d in itr.db_data.user_data])
        max_size = np.max([d.matrix_size for d in itr.db_data.user_data])
        return [i, max_size, max_cond]

    xy_binned = miplot.computeIterationStats(runs,
                                             work_bins=len(runs[0].iters),
                                             filteritr=miplot.filteritr_all,
                                             fnItrStats=fnItrStats,
                                             arr_fnAgg=[np.mean,
                                                        np.mean,
                                                        np.mean])

    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    if which == 'cond':
        ax.set_ylabel('Condition')
        line, = ax.plot(xy_binned[:, 0], xy_binned[:, 2],
                       *args, **kwargs)
    else:
        ax.set_ylabel('Matrix Size')
        line, = ax.plot(xy_binned[:, 0], xy_binned[:, 1], *args,
                        **kwargs)

    return line.get_xydata(), [line]

def plot_all(runs, **kwargs):
    filteritr = kwargs.pop("filteritr", miplot.filteritr_all)
    modifier = kwargs.pop("modifier", 1.)
    TOLs_count = len(np.unique([itr.TOL for _, itr
                                in miplot.enum_iter(runs, filteritr)]))
    convergent_count = len([itr.TOL for _, itr
                            in miplot.enum_iter(runs, miplot.filteritr_convergent)])
    iters_count = np.sum([len(r.iters) for r in runs])
    verbose = kwargs.pop('verbose', False)
    legend_outside = kwargs.pop("legend_outside", 5)
    if verbose:
        def print_msg(*args):
            print(*args)
    else:
        def print_msg(*args):
            return

    fnNorm = kwargs.pop("fnNorm", None)

    figures = []
    def add_fig():
        figures.append(plt.figure())
        return figures[-1].gca()

    label_fmt = '{label}'
    Ref_kwargs = {'ls': '--', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}
    print_msg("plotWorkVsMaxError")
    ax = add_fig()
    Ref_kwargs = {'ls': '--', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}
    ErrEst_kwargs = {'fmt': '--*','label': label_fmt.format(label='Error Estimate')}
    Ref_ErrEst_kwargs = {'ls': '-.', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}
    try:
        miplot.plotWorkVsMaxError(ax, runs,
                                  fnWork=lambda run, i: run.iters[i].calcTotalWork(),
                                  filteritr=filteritr,
                                  modifier=modifier, fmt='-*',
                                  work_spacing=np.sqrt(2),
                                  Ref_kwargs=Ref_kwargs)
        ax.set_xlabel('Avg. Iteration Work')
    except:
        miplot.plot_failed(ax)

    print_msg("plotWorkVsMaxError")
    ax = add_fig()
    try:
        miplot.plotWorkVsMaxError(ax, runs, filteritr=filteritr,
                                  fnWork=lambda run, i:
                                  run.iters[i].calcTotalTime(),
                                  modifier=modifier, fmt='-*',
                                  work_spacing=np.sqrt(2),
                                  Ref_kwargs=Ref_kwargs)
        ax.set_xlabel('Avg. Iteration Time')
    except:
        miplot.plot_failed(ax)

    print_msg("plotSeeds")
    try:
        ax = add_fig()
        plotSeeds(ax, runs, '-o', fnNorm=fnNorm,
                  label='Last iteration', Ref_kwargs=Ref_kwargs)
        plotSeeds(ax, runs, '-o', fnNorm=fnNorm,
                  Ref_kwargs=None,
                  iter_idx=int(len(runs[0].iters)/4))
    except:
        miplot.plot_failed(ax)

    print_msg("plotUserData")
    ax = add_fig()
    try:
        plotUserData(ax, runs, '-o', which='cond')
    except:
        miplot.plot_failed(ax)

    ax = add_fig()
    try:
        plotUserData(ax, runs, '-o', which='size')
    except:
        miplot.plot_failed(ax)

    print_msg("plotDirections")
    ax = add_fig()
    try:
        miplot.plotDirections(ax, runs, miplot.plotExpectVsLvls,
                              fnNorm=fnNorm)
    except:
        miplot.plot_failed(ax)

    print_msg("plotDirections")
    ax = add_fig()
    try:
        miplot.plotDirections(ax, runs, miplot.plotWorkVsLvls,
                              fnNorm=fnNorm)
    except:
        miplot.plot_failed(ax)

    if runs[0].params.min_dim > 0 and runs[0].last_itr.lvls_max_dim() > 2:
        run = runs[0]
        from mimclib import setutil
        if run.params.qoi_example == 'sf-matern':
            profit_calc = setutil.MIProfCalculator([0.0] * run.params.min_dim,
                                                   run.params.miproj_set_xi,
                                                   run.params.miproj_set_sexp,
                                                   run.params.miproj_set_mul)
        else:
            qoi_N = run.params.miproj_max_var
            td_w = [0.] * (run.params.min_dim + qoi_N)
            ft_w = [0.] * run.params.min_dim + [1.] * qoi_N
            profit_calc = setutil.TDFTProfCalculator(td_w, ft_w)

        profits = run.last_itr._lvls.calc_log_prof(profit_calc)
        reduced_run = runs[0].reduceDims(np.arange(0, runs[0].params.min_dim),
                                         profits)    # Keep only the spatial dimensions
        print_msg("plotDirections")
        ax = add_fig()
        try:
            miplot.plotDirections(ax, [reduced_run],
                                  miplot.plotExpectVsLvls, fnNorm=fnNorm,
                                  x_axis='ell')
        except:
            miplot.plot_failed(ax)
        print_msg("plotDirections")
        ax = add_fig()
        try:
            miplot.plotDirections(ax, [reduced_run],
                                  miplot.plotWorkVsLvls, fnNorm=fnNorm,
                                  x_axis='ell')
        except:
            miplot.plot_failed(ax)
    print_msg("plotBestNTerm")
    try:
        ax = add_fig()
        plotBestNTerm(ax, runs, '-o', Ref_kwargs=Ref_kwargs)
    except:
        miplot.plot_failed(ax)

    print_msg("plotWorkVsLvlStats")
    ax = add_fig()
    try:
        miplot.plotWorkVsLvlStats(ax, runs, '-ob',
                                  filteritr=filteritr,
                                  label=label_fmt.format(label='Total dim.'),
                                  active_kwargs={'fmt': '-*g', 'label':
                                                 label_fmt.format(label='Max active dim.')},
                                  maxrefine_kwargs={'fmt': '-sr', 'label':
                                                    label_fmt.format(label='Max refinement')})
    except:
        miplot.plot_failed(ax)

    figures.extend(plotSingleLevel(runs,
                                   kwargs['input_args'],
                                   modifier=modifier,
                                   fnNorm=fnNorm,
                                   Ref_kwargs=Ref_kwargs))

    for fig in figures:
        for ax in fig.axes:
            miplot.add_legend(ax, outside=legend_outside)
    return figures

def plotSingleLevel(runs, input_args, *args, **kwargs):
    modifier = kwargs.pop('modifier', None)
    fnNorm = kwargs.pop('fnNorm', None)
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    plotIndividual  = kwargs.pop('plot_individual', True)
    from mimclib import db as mimcdb
    db = mimcdb.MIMCDatabase(**input_args.db_args)
    print("Reading data")

    db_tag = input_args.db_tag
    adaptive_runs = False
    if db_tag.endswith('-adapt'):
        adaptive_runs = True
        db_tag = db_tag[:-len('-adapt')]

    fig_W = plt.figure()
    fig_T = plt.figure()
    fig_Tc = plt.figure()
    fix_runs = []
    while True:
        fix_tag = db_tag + "-fix-" + str(len(fix_runs))
        run_data = db.readRuns(tag=fix_tag, done_flag=input_args.done_flag)
        if len(run_data) == 0:
            break
        print("Got", fix_tag)
        assert(len(run_data) == 1)
        # Modify work estimates to account for space discretization
        # TODO: WARNING: Different behavior between poisson and matern
        # ell = len(fix_runs)
        #work_per_sample = run_data[0].params.beta ** (run_data[0].params.gamma * ell)  # matern
        # work_per_sample = run_data[0].params.beta ** (run_data[0].params.gamma * ell/2.0) # poisson
        # for itr in run_data[0].iters:
        #     itr.tW *= work_per_sample
        #     itr.Wl_estimate = itr.tW
        fix_runs.append(run_data[0])

    fnWork = lambda run, i: run.iters[i].calcTotalWork()
    if runs[0].params.miproj_reuse_samples:
        fnTime = lambda run, i: run.iter_total_times[i]
    else:
        fnTime = lambda run, i: run.iters[i].totalTime #run.iter_total_times[i]
    fnTime_calc = lambda run, i: run.iters[i].calcTotalTime()

    work_bins = 50
    work_spacing = np.sqrt(2)
    if adaptive_runs:
        runs_adaptive = runs
        runs_priori = db.readRuns(tag=db_tag, done_flag=input_args.done_flag)
    else:
        runs_adaptive = db.readRuns(tag=db_tag + "-adapt", done_flag=input_args.done_flag)
        runs_priori = runs

    if input_args.qoi_exact is not None:
        print("Setting errors")
        fnExactErr = lambda itrs, e=input_args.qoi_exact: \
                     fnNorm([v.calcEg() + e*-1 for v in itrs])
        miplot.set_exact_errors(fix_runs + runs_adaptive, fnExactErr)

    if plotIndividual:
        for i, rr in enumerate(fix_runs):
            miplot.plotWorkVsMaxError(fig_W.gca(), [rr],
                                      modifier=modifier, fnWork=fnWork,
                                      fnAggError=np.min, fmt='--x',
                                      work_bins=1000, Ref_kwargs=None,
                                      label='\\ell={}'.format(i), alpha=0.7)
            miplot.plotWorkVsMaxError(fig_T.gca(), [rr],
                                      fnWork=fnTime,
                                      modifier=modifier, fmt='--x',
                                      fnAggError=np.min,
                                      work_bins=1000, Ref_kwargs=None,
                                      label='\\ell={}'.format(i),
                                      alpha=0.7)
            miplot.plotWorkVsMaxError(fig_Tc.gca(), [rr],
                                      fnWork=fnTime_calc,
                                      modifier=modifier, fmt='--x',
                                      fnAggError=np.min,
                                      work_bins=1000, Ref_kwargs=None,
                                      label='\\ell={}'.format(i),
                                      alpha=0.7)


    for rr, label in [[fix_runs, 'SL'],
                       [runs_adaptive, 'ML Adaptive'], [runs_priori, 'ML']]:
        if rr is None or len(rr) == 0:
            continue
        miplot.plotWorkVsMaxError(fig_W.gca(), rr,
                                  modifier=modifier, fnWork=fnWork,
                                  fnAggError=np.min, fmt='-*',
                                  work_bins=work_bins,
                                  Ref_kwargs=Ref_kwargs if rr==runs else None,
                                  work_spacing=work_spacing, label=label)
        miplot.plotWorkVsMaxError(fig_T.gca(), rr, fnWork=fnTime,
                                  modifier=modifier, fmt='-*',
                                  fnAggError=np.min,
                                  work_bins=work_bins,
                                  Ref_kwargs=Ref_kwargs if rr==runs else None,
                                  work_spacing=work_spacing, label=label)
        miplot.plotWorkVsMaxError(fig_Tc.gca(), rr, fnWork=fnTime_calc,
                                  modifier=modifier, fmt='-*',
                                  fnAggError=np.min,
                                  work_bins=work_bins,
                                  Ref_kwargs=Ref_kwargs if rr==runs else None,
                                  work_spacing=work_spacing, label=label)

    fig_W.gca().set_xlabel('Avg. Iteration Work')
    fig_T.gca().set_xlabel('Avg. Iteration Time (tic/toc)')
    fig_Tc.gca().set_xlabel('Avg. Iteration Time (calculated)')
    return [fig_W, fig_T, fig_Tc]

if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()
    from mimclib.plot import run_plot_program
    run_plot_program(plot_all)
