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
    print_msg("plotWorkVsMaxError")
    ax = add_fig()
    Ref_kwargs = {'ls': '--', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}
    ErrEst_kwargs = {'fmt': '--*','label': label_fmt.format(label='Error Estimate')}
    Ref_ErrEst_kwargs = {'ls': '-.', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}
    try:
        miplot.plotWorkVsMaxError(ax, runs,
                                  iter_stats_args=dict(work_spacing=np.log(np.sqrt(2)),
                                                       filteritr=filteritr),
                                  fnWork=lambda run, i:
                                  run.iters[i].calcTotalWork(),
                                  modifier=modifier, fmt='-*',
                                  Ref_kwargs=Ref_kwargs)
        ax.set_xlabel('Avg. Iteration Work')
    except:
        miplot.plot_failed(ax)

    print_msg("plotWorkVsMaxError")
    ax = add_fig()
    try:
        miplot.plotWorkVsMaxError(ax, runs,
                                  iter_stats_args=dict(work_spacing=np.log(np.sqrt(2)),
                                                       filteritr=filteritr),
                                  fnWork=lambda run, i:
                                  run.iters[i].calcTotalTime(),
                                  modifier=modifier, fmt='-*',
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
            qoi_N = run.params.miproj_max_vars
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

    flip = True
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

    def filter_dec(xy):
        xy = xy[xy[:, 1].argsort(), :]
        while True:
            sel = np.ones(len(xy), dtype=np.bool)
            desel = np.diff(xy[:, 2]) > 0
            if not np.any(desel):
                break
            sel[1:][desel] = False
            xy = xy[sel, :]
        return xy

    iter_stats_args = dict(work_bins=1000)
    if plotIndividual:
        for i, rr in enumerate(fix_runs):
            miplot.plotWorkVsMaxError(fig_W.gca(), [rr],
                                      flip=flip,
                                      iter_stats_args=iter_stats_args,
                                      modifier=modifier, fnWork=fnWork,
                                      fnAggError=np.min, fmt='--x',
                                      label='\\ell={}'.format(i), alpha=0.7)
            miplot.plotWorkVsMaxError(fig_T.gca(), [rr],
                                      flip=flip,
                                      iter_stats_args=iter_stats_args,
                                      fnWork=fnTime,
                                      modifier=modifier, fmt='--x',
                                      fnAggError=np.min,
                                      label='\\ell={}'.format(i),
                                      alpha=0.7)
            miplot.plotWorkVsMaxError(fig_Tc.gca(), [rr],
                                      flip=flip,
                                      iter_stats_args=iter_stats_args,
                                      fnWork=fnTime_calc,
                                      modifier=modifier, fmt='--x',
                                      fnAggError=np.min,
                                      label='\\ell={}'.format(i),
                                      alpha=0.7)

    rates_ML, rates_SL = None, None
    if runs[0].params.qoi_example == 'sf-kink':
        N = runs[0].params.miproj_max_vars
        if N < 3:
            rates_ML = [-1., 1, 1.]
        elif N == 3:
            rates_ML = [-1., 1, 4.]
        else:
            rates_ML = [-3., N, 3.]
        rates_SL = [-3., 3. + N, 1.]

    for rr, label, rates, ref_ls in [[fix_runs, 'SL', rates_SL, '-.'],
                                     [runs_adaptive, 'ML Adaptive', rates_ML if
                                      runs == runs_adaptive else None, '--'],
                                     [runs_priori, 'ML', rates_ML if runs ==
                                      runs_priori else None, '--']]:
        if rr is None or len(rr) == 0:
            continue
        if rr == fix_runs:
            iter_stats_args = dict(work_bins=1000,
                                   work_spacing=None,
                                   fnFilterData=filter_dec)
        else:
            iter_stats_args = dict(work_bins=work_bins,
                                   work_spacing=work_spacing,
                                   fnFilterData=filter_dec)

        if rates is not None:
            if rates[1] == 1:
                if rates[0] == 1:
                    base = r'W'
                else:
                    base = r'W^{{{:.2g}}}'.format(rates[0])
            else:
                base = r'W^{{\frac{{ {:.2g} }}{{ {:.2g} }}}}'.format(rates[0], rates[1])

            if rates[2] == 1:
                log_factor = r'|\log(W)|'
            else:
                log_factor = r'|\log(W)|^{{{:.2g}}}'.format(rates[2])

            Ref_kwargs['label'] = '${}{}$'.format(base, log_factor)
            Ref_kwargs['ls'] = ref_ls

        for fig, curFnWork in [[fig_W, fnWork], [fig_T, fnTime],
                               [fig_Tc, fnTime_calc]]:
            data, _ = miplot.plotWorkVsMaxError(fig.gca(), rr,
                                                flip=True,
                                                modifier=modifier,
                                                fnWork=curFnWork,
                                                fnAggError=np.min,
                                                fmt='-*',
                                                iter_stats_args=iter_stats_args,
                                                Ref_kwargs=Ref_kwargs
                                                if rates is None and
                                                rr == runs else None,
                                                label=label)
            data = data[np.argsort(data[:, 0]), :]
            if rates is not None:
                def fnRate(x, rr=rates):
                    base = np.min(x)
                    return (1+x/base)**(rr[0]/rr[1])*np.abs(np.log(1+x/base)**rr[2])
                fig.gca().add_line(miplot.FunctionLine2D(fn=fnRate,
                                                         data=data[-len(data)//3:, :],
                                                         **Ref_kwargs))

    if flip:
        fig_W.gca().set_ylabel('Avg. Iteration Work')
        fig_T.gca().set_ylabel('Avg. Iteration Time (tic/toc)')
        fig_Tc.gca().set_ylabel('Avg. Iteration Time (calculated)')
    else:
        fig_W.gca().set_xlabel('Avg. Iteration Work')
        fig_T.gca().set_xlabel('Avg. Iteration Time (tic/toc)')
        fig_Tc.gca().set_xlabel('Avg. Iteration Time (calculated)')

    return [fig_W, fig_T, fig_Tc]

if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()
    from mimclib.plot import run_plot_program
    run_plot_program(plot_all)
