#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
mpl.rc('text', usetex=True)
mpl.rc('font', **{'family': 'normal', 'weight': 'demibold',
                  'size': 15})
mpl.rc('lines', linewidth=4, markersize=10, markeredgewidth=1.)
mpl.rc('markers', fillstyle='none')
mpl.rc('axes', labelsize=20,)

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
                                                  linewidth=2,
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
                                                  linewidth=2,
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
    modifier = kwargs.pop("modifier", None)
    TOLs_count = len(np.unique([itr.TOL for _, itr
                                in miplot.enum_iter(runs, filteritr)]))
    convergent_count = len([itr.TOL for _, itr
                            in miplot.enum_iter(runs, miplot.filteritr_convergent)])
    iters_count = np.sum([len(r.iters) for r in runs])
    verbose = kwargs.pop('verbose', False)
    legend_outside = kwargs.pop("legend_outside", 8)
    if verbose:
        def print_msg(*args):
            print(*args)
    else:
        def print_msg(*args):
            return

    fnNorm = kwargs.pop("fnNorm", None)

    figures = []
    def add_fig(name):
        figures.append(plt.figure())
        figures[-1].label = "fig:" + name
        figures[-1].file_name = name
        return figures[-1].gca()

    label_fmt = '{label}'
    print_msg("plotWorkVsMaxError")
    ax = add_fig('work-vs-max-error')
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
        raise

    print_msg("plotWorkVsMaxError")
    ax = add_fig('time-vs-max-error')
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
        ax = add_fig('error-vs-dim')
        plotSeeds(ax, runs, '-o', fnNorm=fnNorm,
                  label='Last iteration', Ref_kwargs=Ref_kwargs)
        plotSeeds(ax, runs, '-o', fnNorm=fnNorm,
                  Ref_kwargs=None,
                  iter_idx=int(len(runs[0].iters)/4))
    except:
        miplot.plot_failed(ax)

    print_msg("plotUserData")
    ax = add_fig('cond-vs-iteration')
    try:
        plotUserData(ax, runs, '-o', which='cond')
    except:
        miplot.plot_failed(ax)

    ax = add_fig('size-vs-iteration')
    try:
        plotUserData(ax, runs, '-o', which='size')
    except:
        miplot.plot_failed(ax)

    print_msg("plotDirections")
    ax = add_fig('error-vs-lvl')
    #try:
    miplot.plotDirections(ax, runs, miplot.plotExpectVsLvls,
                          fnNorm=fnNorm,
                          dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'log_ell'}])
    # except:
    #     miplot.plot_failed(ax)
    #     raise

    print_msg("plotDirections")
    ax = add_fig('work-vs-lvl')
    try:
        miplot.plotDirections(ax, runs, miplot.plotWorkVsLvls,
                              fnNorm=fnNorm, dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'log_ell'}])
    except:
        miplot.plot_failed(ax)
        raise

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
            miproj_set_dexp = run.params.miproj_set_dexp if run.params.min_dim > 0 else 0
            td_w = [miproj_set_dexp] * run.params.min_dim + [0.] * qoi_N
            hc_w = [0.] * run.params.min_dim +  [run.params.miproj_set_sexp] * qoi_N
            profit_calc = setutil.TDHCProfCalculator(td_w, hc_w)

        profits = run.last_itr._lvls.calc_log_prof(profit_calc)
        reduced_run = runs[0].reduceDims(np.arange(0, runs[0].params.min_dim),
                                         profits)    # Keep only the spatial dimensions
        print_msg("plotDirections")
        ax = add_fig('reduced-expect-vs-lvl')
        try:
            miplot.plotDirections(ax, [reduced_run],
                                  miplot.plotExpectVsLvls, fnNorm=fnNorm,
                                  dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'log_ell'}])
        except:
            miplot.plot_failed(ax)
        print_msg("plotDirections")
        ax = add_fig('reduced-work-vs-lvl')
        try:
            miplot.plotDirections(ax, [reduced_run],
                                  miplot.plotWorkVsLvls, fnNorm=fnNorm,
                                  dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'log_ell'}])
        except:
            miplot.plot_failed(ax)
    print_msg("plotBestNTerm")
    try:
        ax = add_fig('best-nterm')
        plotBestNTerm(ax, runs, '-o', Ref_kwargs=Ref_kwargs)
    except:
        miplot.plot_failed(ax)

    print_msg("plotWorkVsLvlStats")
    ax = add_fig('stats-vs-lvls')
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
            legend = miplot.add_legend(ax, outside=legend_outside,
                                       frameon=False, loc='best')
            if legend is not None:
                legend.get_frame().set_facecolor('none')
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

    figures = []
    def add_fig(name):
        figures.append(plt.figure())
        figures[-1].label = "fig:" + name
        figures[-1].file_name = name
        return figures[-1].gca()

    fig_W = add_fig('errors-vs-work')
    fig_T = add_fig('times-vs-work')
    fig_Tc = add_fig('total-times-vs-work')
    fix_runs = []
    while True:
        fix_tag = db_tag + "-fix-" + str(len(fix_runs))
        run_data = db.readRuns(tag=fix_tag, done_flag=input_args.done_flag)
        if len(run_data) == 0:
            print("Couldn't get", fix_tag, input_args.done_flag)
            break
        print("Got", fix_tag)
        assert(len(run_data) == 1)
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

    if len(runs_priori) > 0:
        print("N", runs_priori[0].params.miproj_max_vars,
              "DEXP", runs_priori[0].params.miproj_set_dexp)

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
            miplot.plotWorkVsMaxError(fig_W, [rr],
                                      flip=flip,
                                      iter_stats_args=iter_stats_args,
                                      modifier=modifier, fnWork=fnWork,
                                      fnAggError=np.min, fmt=':xk',
                                      linewidth=2, markersize=4,
                                      #label='\\ell={}'.format(i),
                                      alpha=0.4)
            miplot.plotWorkVsMaxError(fig_T, [rr],
                                      flip=flip,
                                      iter_stats_args=iter_stats_args,
                                      fnWork=fnTime,
                                      modifier=modifier, fmt=':xk',
                                      fnAggError=np.min,
                                      linewidth=2, markersize=4,
                                      #label='\\ell={}'.format(i),
                                      alpha=0.4)
            miplot.plotWorkVsMaxError(fig_Tc, [rr],
                                      flip=flip,
                                      iter_stats_args=iter_stats_args,
                                      fnWork=fnTime_calc,
                                      modifier=modifier, fmt=':xk',
                                      fnAggError=np.min,
                                      linewidth=2, markersize=4,
                                      #label='\\ell={}'.format(i),
                                      alpha=0.4)

    rates_ML, rates_SL = None, None
    if runs[0].params.qoi_example == 'sf-kink':
        t = 3.
        N = runs[0].params.miproj_max_vars
        alpha_r = [t, N]
        alpha = alpha_r[0]/alpha_r[1]
        gamma = 1
        beta = 1

        if gamma/beta <= 1/alpha:
            rates_ML = [alpha_r[1], alpha_r[0], 0]
        else:
            rates_ML = [gamma, beta, 0]

        if gamma/beta < 1/alpha:
            rates_ML[-1] = 2
        elif gamma/beta == 1/alpha:
            rates_ML[-1] = 3 + 1/alpha
        else:
            rates_ML[-1] = 1
        rates_SL = [alpha_r[1]*beta + alpha_r[0]*gamma, alpha_r[0]*beta, 1.]

    from fractions import gcd
    g = gcd(rates_ML[0], rates_ML[1])
    rates_ML[0] /= g
    rates_ML[1] /= g
    g = gcd(rates_SL[0], rates_SL[1])
    rates_SL[0] /= g
    rates_SL[1] /= g

    for zorder, rr, label, rates, ref_ls in [[10,fix_runs, 'SL', rates_SL, '-.'],
                                             [11,runs_priori, 'ML', rates_ML if runs ==
                                              runs_priori else None, '--'],
                                             [12,runs_adaptive, 'ML Adaptive', rates_ML if
                                              runs == runs_adaptive else None, '--']]:
        if rr is None or len(rr) == 0:
            continue
        if rr != fix_runs:
            iter_stats_args = dict(work_bins=1000,
                                   work_spacing=None,
                                   fnFilterData=None)
        else:
            iter_stats_args = dict(work_bins=1000,
                                   work_spacing=None,
                                   fnFilterData=filter_dec)

        if rates is not None:
            if rates[1] == 1:  # Denominator
                if rates[0] == 1:  # Numerator
                    base = r'\epsilon^{-1}'
                else:
                    base = r'\epsilon^{{-{:.2g}}}'.format(rates[0])
            else:
                base = r'\epsilon^{{-\frac{{ {:.2g} }}{{ {:.2g} }}}}'.format(rates[0], rates[1])

            if rates[2] == 0:
                log_factor = r''
            elif rates[2] == 1:
                log_factor = r'\log(\epsilon^{-1})'
            else:
                log_factor = r'\log(\epsilon^{{-1}})^{{{:.2g}}}'.format(rates[2])

            Ref_kwargs['label'] = '${}{}$'.format(base, log_factor)
            Ref_kwargs['ls'] = ref_ls

        for fig, curFnWork in [[fig_W, fnWork], [fig_T, fnTime],
                               [fig_Tc, fnTime_calc]]:
            data, _ = miplot.plotWorkVsMaxError(fig, rr,
                                                flip=True,
                                                modifier=modifier,
                                                fnWork=curFnWork,
                                                fnAggError=np.min,
                                                fmt='-',
                                                iter_stats_args=iter_stats_args,
                                                Ref_kwargs=Ref_kwargs
                                                if rates is None and
                                                rr == runs else None,
                                                zorder=zorder,
                                                label=label)
            data = data[np.argsort(data[:, 0]), :]
            if rates is not None:
                def fnRate(x, rr=rates):
                    return (x)**(-rr[0]/rr[1])*np.abs(np.log(x)**rr[2])
                fig.add_line(miplot.FunctionLine2D(fn=fnRate,
                                                         linewidth=2,
                                                         zorder=5,
                                                         data=data[:len(data)//3, :],
                                                         **Ref_kwargs))

    if flip:
        fig_W.set_ylabel('Work Estimate')
        fig_T.set_ylabel('Time, total [s.]')
        fig_Tc.set_ylabel('Time [s.]')
    else:
        fig_W.set_xlabel('Work Estimate')
        fig_T.set_xlabel('Time, total [s.]')
        fig_T.set_xlabel('Time [s.]')

    return figures

if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()
    from mimclib.plot import run_plot_program
    run_plot_program(plot_all)
