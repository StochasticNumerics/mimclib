from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import matplotlib
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pylab as plt
from . import mimc
from matplotlib.ticker import MaxNLocator
import os
from matplotlib2tikz import save as tikz_save
from . import db as mimcdb
from . import test
import argparse
import warnings
import itertools
from mimclib import Bunch
from scipy.stats import norm
import traceback

__all__ = []

def save_pdf(out, figures, close=True):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            if close:
                plt.close(fig)


def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]

    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
                  return_index=return_index,
                  return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

def _scatter(ax, x, y, *args, **kwargs):
    xy = unique_rows(np.vstack((x,y)).transpose())
    return ax.scatter(xy[:, 0], xy[:, 1], *args, **kwargs)

def public(sym):
    __all__.append(sym.__name__)
    return sym

def _format_latex_sci(x):
    ss = '{:.2e}'.format(x).lower()
    if "e" not in ss:
        return ss
    parts = [float(s) for s in ss.split('e')]
    assert(len(parts) == 2)
    return "{:.2g} \\times 10^{{ {:g} }}".format(parts[0], parts[1])

def _formatPower(rate):
    rate = "{:.2g}".format(rate)
    if rate == "-1":
        return "-"
    elif rate == "1":
        return ""
    return rate

try:
    raise Exception()
    import statsmodels.api as sm
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore")
        def ratefit(x, y):
            f = sm.RLM(y, sm.add_constant(x)).fit()
            return f.params[1], f.params[0]
except:
    def ratefit(x, y):
        # This should remove outlires on its own
        while True:
            c = np.polyfit(x, y, 1)
            err = np.abs(y-np.polyval(c, x))
            sigma = np.std(err)
            sel = err <= sigma
            if np.all(sel) or np.sum(sel) < 2:
                break
            x, y = x[sel], y[sel]
        return c[0], c[1]

@public
class FunctionLine2D(plt.Line2D):
    def __init__(self, *args, **kwargs):
        self.flip = kwargs.pop('flip', False)
        self.orig_fn = fn = kwargs.pop('fn')
        self.fn = self.orig_fn
        data = kwargs.pop('data', None)
        log_data = kwargs.pop('log_data', True)
        if data is not None:
            self.set_data(data, log_data)

        super(FunctionLine2D, self).__init__([], [], *args, **kwargs)

    def _linspace(self, lim, scale, N=100):
        if scale == 'log':
            return np.exp(np.linspace(np.log(lim[0]), np.log(lim[1]), N))
        else:
            return np.linspace(lim[0], lim[1], N)

    def set_data(self, data, log_data=True):
        x = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        if len(x) > 0 and len(y) > 0:
            if log_data:
                consts = [np.mean(y/self.orig_fn(x)), 0]
                #consts = ratefit(self.orig_fn(x), y)
            else:
                consts = [1., np.mean(y-self.orig_fn(x))]
            self.fn = lambda x, cc=consts, ff=self.orig_fn: cc[0] * ff(x) + cc[1]


    def draw(self, renderer):
        ax = self.axes
        if self.flip:
            y = self._linspace(ax.get_ylim(), ax.get_yscale())
            self.set_xdata(self.fn(y))
            self.set_ydata(y)
        else:
            x = self._linspace(ax.get_xlim(), ax.get_xscale())
            self.set_xdata(x)
            self.set_ydata(self.fn(x))
        plt.Line2D.draw(self, renderer)

    @staticmethod
    def ExpLine(*args, **kwargs):
        rate = kwargs.pop('rate', None)
        log_data = kwargs.pop('log_data', True)
        data = kwargs.pop('data', None)
        const = kwargs.pop('const', 1)
        if rate is None:
            assert data is not None and len(data) > 0, "rate or data must be given"
            x = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            if len(x) > 0 and len(y) > 0:
                if log_data:
                    rate, const = ratefit(np.log(x), np.log(y))
                    const = np.exp(const)
                else:
                    rate, const = ratefit(x, y)
            data = None

        if "label" in kwargs:
            kwargs["label"] = kwargs.pop("label").format(rate=rate)
        def fnExp(x, r=rate, cc=const):
            with np.errstate(divide='ignore', invalid='ignore'):
                return cc*np.array(x)**r;
        return FunctionLine2D(*args, fn=fnExp, data=data,
                              log_data=log_data, **kwargs)


class StepFunction(object):
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array-like
    y : array-like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import StepFunction
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print f(3.2)
    3.0
    >>> print f([[3.2,4.5],[24,-3.1]])
    [[  3.   4.]
     [ 19.   0.]]
    >>> f2 = StepFunction(x, y, side='right')
    >>>
    >>> print f(3.0)
    2.0
    >>> print f2(3.0)
    3.0
    """

    def __init__(self, x, y, ival=0., sorted=False, side='left'):

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):
        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


class ECDF(StepFunction):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array-like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Empirical CDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import ECDF
    >>>
    >>> ecdf = ECDF([3, 3, 1, 4])
    >>>
    >>> ecdf([3, 55, 0.5, 1.5])
    array([ 0.75,  1.  ,  0.  ,  0.25])
    """

    def __init__(self, x, side='right'):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1. / nobs, 1, nobs)
        super(ECDF, self).__init__(x, y, side=side, sorted=True)


def __get_stats(data, groupby=0, staton=1):
    data = sorted(data, key=lambda xx: xx[groupby])
    x = []
    y = []
    for k, itr in itertools.groupby(data, key=lambda xx: xx[groupby]):
        all_y = [d[staton] for d in itr]
        y.append([np.nanpercentile(all_y, 5),
                  np.nanpercentile(all_y, 50),
                  np.nanpercentile(all_y, 95)])
        x.append(k)
    return np.array(x), np.array(y)


def __calc_moments(runs, seed=None, direction=None, fnNorm=None):
    if seed is None and direction is None:
        direction = np.array([1]) #  Revert to 1-dim

    dim = len(seed) if seed is not None else len(direction)
    seed = np.array(seed) if seed is not None else np.zeros(dim, dtype=np.uint32)
    direction = np.array(direction) if direction is not None else np.ones(dim, dtype=np.uint32)
    moments = runs[0].last_itr.psums_delta.shape[1]
    psums_delta, psums_fine, Tl, Wl, Vl_estimate, M = [None]*6
    for i, curRun in enumerate(runs):
        cur = seed
        inds = []
        while True:
            ii = curRun.last_itr.lvls_find(cur)
            if ii is None:
                break
            inds.append(ii)
            cur = cur + direction

        L = len(inds)
        if psums_delta is None:
            psums_delta = curRun.last_itr.psums_delta[inds]
            psums_fine = curRun.last_itr.psums_fine[inds]
            M = curRun.last_itr.M[inds]
            Vl_estimate = np.zeros((L, len(runs)))
            Tl = np.zeros((L, len(runs)))
            Wl = np.zeros((L, len(runs)))
            Vl_estimate[:, i] = curRun.Vl_estimate[inds]
            Tl[:, i] = curRun.last_itr.tT[inds]/curRun.last_itr.M[inds]
            Wl[:, i] = curRun.last_itr.tW[inds]/curRun.last_itr.M[inds]
            continue

        oldL = np.minimum(L, len(M))
        psums_delta[:oldL] += curRun.last_itr.psums_delta[inds[:oldL]]
        psums_fine[:oldL] += curRun.last_itr.psums_fine[inds[:oldL]]
        M[:oldL] += curRun.last_itr.M[inds[:oldL]]

        if L > oldL:
            psums_delta = np.append(psums_delta,
                                    curRun.last_itr.psums_delta[inds[oldL:]], axis=0)
            psums_fine = np.append(psums_fine,
                                   curRun.last_itr.psums_fine[inds[oldL:]], axis=0)
            M = np.append(M, curRun.last_itr.M[inds[oldL:]], axis=0)

            tmp = Vl_estimate.shape[0]
            Vl_estimate.resize((L, len(runs)), refcheck=False)
            Vl_estimate[tmp:, :] = np.nan

            tmp = Tl.shape[0]
            Tl.resize((L, len(runs)), refcheck=False)
            Tl[tmp:, :] = np.nan
            tmp = Wl.shape[0]
            Wl.resize((L, len(runs)), refcheck=False)
            Wl[tmp:, :] = np.nan

        Vl_estimate[:L, i] = curRun.Vl_estimate[inds]
        Vl_estimate[(L+1):, i] = np.nan
        Tl[:L, i] = curRun.last_itr.tT[inds]/curRun.last_itr.M[inds]
        Tl[L:, i] = np.nan
        Wl[:L, i] = curRun.last_itr.tW[inds]/curRun.last_itr.M[inds]
        Wl[L:, i] = np.nan

    central_delta_moments = np.empty((psums_delta.shape[0],
                                      psums_delta.shape[1]), dtype=float)
    central_fine_moments = np.empty((psums_fine.shape[0],
                                     psums_fine.shape[1]), dtype=float)
    for m in range(1, psums_delta.shape[1]+1):
        central_delta_moments[:, m-1] = fnNorm(mimc.compute_central_moment(psums_delta, M, m))
        central_fine_moments[:, m-1] = fnNorm(mimc.compute_central_moment(psums_fine, M, m))

    ret = Bunch()
    ret.central_delta_moments = central_delta_moments
    ret.central_fine_moments = central_fine_moments
    ret.Vl_estimate = Vl_estimate
    ret.Tl = Tl
    ret.Wl = Wl
    ret.M = M
    return ret

@public
def normalize_fmt(args, kwargs):
    if "fmt" in kwargs:        # Normalize behavior of errorbar() and plot()
        args = (kwargs.pop('fmt'), ) + args
    return args, kwargs


@public
def filteritr_last(run, iter_idx):
    return len(run.iters)-1 == iter_idx


@public
def filteritr_convergent(run, iter_idx):
    return run.iters[iter_idx].total_error_est <= run.iters[iter_idx].TOL

@public
def filteritr_all(run, iter_idx):
    return True


@public
def enum_iter(runs, fnFilter=filteritr_all):
    for r in runs:
        for i in xrange(0, len(r.iters)):
            if fnFilter(r, i):
                yield r, r.iters[i]


@public
def enum_iter_i(runs, fnFilter=filteritr_all):
    for r in runs:
        for i in xrange(0, len(r.iters)):
            if fnFilter(r, i):
                yield i, r, r.iters[i]

@public
def computeIterationStats(runs, fnItrStats, arr_fnAgg, work_bins=50,
                          filteritr=filteritr_all, work_spacing=None,
                          fnFilterData=None):
    xy = []
    for r in runs:
        prev = 0
        for i in xrange(0, len(r.iters)):
            if not filteritr(r, i):
                continue
            stats = fnItrStats(r, i)
            assert(len(stats) == len(arr_fnAgg))
            stats = [stats[i] if stats[i] is not None or len(xy)==0
                     else xy[-1][i] for i in xrange(len(stats))]
            if not np.isfinite(stats[0]):
                continue
            xy.append(stats)
    xy = np.array(xy)
    if fnFilterData is not None:
        xy = fnFilterData(xy)
    if len(xy) == 0:
        return xy
    bins = np.digitize(xy[:, 0], np.linspace(np.min(xy[:, 0]),
                                             np.max(xy[:, 0]), work_bins))
    bins[bins == work_bins] = work_bins-1
    ubins = np.unique(bins)
    xy_binned = np.zeros((len(ubins), len(arr_fnAgg)))
    for i, b in enumerate(ubins):
        d = xy[bins==b, :]
        for j in range(0, len(arr_fnAgg)):
            xy_binned[i, j] = arr_fnAgg[j](d[:, j])

    xy_binned = xy_binned[xy_binned[:, 0].argsort(), :]
    if work_spacing is not None:
        sel = np.zeros(xy_binned.shape[0], dtype=np.bool)
        prevWork = xy_binned[0, 0]
        sel[0] = True
        for i in range(0, xy_binned.shape[0]):
            if xy_binned[i, 0] >= prevWork + work_spacing:
                sel[i] = True
                prevWork = xy_binned[i, 0]
        xy_binned = xy_binned[sel, :]
    return xy_binned


@public
def plotDirections(ax, runs, fnPlot,
                   fnNorm=None,
                   plot_fine=False, plot_estimate=False,
                   directions=None, rate=None,
                   dir_kwargs=[{}],
                   label_fmt='{label}', beta=None):
    if directions is None:
        default_max_dims = 5
        max_dim = np.max([np.max(r.last_itr.lvls_max_dim()) for r in runs])
        directions = np.eye(np.minimum(default_max_dims, max_dim), dtype=np.int).tolist()
        cur = np.array(directions[0])
        for i in range(1, len(directions)):
            cur += np.array(directions[i])
            directions.append(cur.tolist())
    add_rates = dict()
    markers = itertools.cycle(['o', 'v', '^', '<', '>', '8', 's', 'p',
                     '*', 'h', 'H', 'D', 'd'])
    linestyles = itertools.cycle(['--', '-.', '-', ':', '-'])
    cycler = ax._get_lines.prop_cycler
    if beta is not None and hasattr(beta, '__iter__'):
        beta = beta[0]

    for j, direction in enumerate(directions):
        mrk = next(markers)
        prop = next(cycler)
        labal = r'$\Delta$' if len(directions)==1 \
                else "$[{}]$".format(",".join(["0" if d==0
                                               else ("" if d==1 else str(d))
                                               +r"\ell" for d in direction]))
        cur_kwargs = {'ax' : ax, 'runs': runs,
                      'linestyle' : '-',
                      'marker' : mrk,
                      'label': label_fmt.format(label=labal),
                      'direction' : direction}
        cur_kwargs.update(prop)
        if plot_fine:
            cur_kwargs['fine_kwargs'] = {'linestyle': '--',
                                         'marker' : mrk}
            cur_kwargs['fine_kwargs'].update(prop)

        if max_dim == 1 and plot_estimate:
            cur_kwargs['estimate_kwargs'] = {'linestyle': ':',
                                             'marker' : mrk,
                                             'label' : label_fmt.format(label='Corrected estimate')}
        cur_kwargs.update(dir_kwargs[np.minimum(j, len(dir_kwargs)-1)])
        line_data, _ = fnPlot(fnNorm=fnNorm, **cur_kwargs)
        if rate is None and len(line_data[1:, :]) > 0:
            # Fit rate
            fit_data = line_data[-3:, :]
            if beta is not None:
                cur_rate = ratefit(np.log(beta) * fit_data[:, 0],
                                   np.log(fit_data[:, 1]))[0]
            else:
                if 'x_axis' not in cur_kwargs or cur_kwargs['x_axis'] == 'ell':
                    cur_rate = ratefit(fit_data[:, 0],
                                       np.log(fit_data[:, 1]))[0]
                else:
                    cur_rate = ratefit(np.log(fit_data[:, 0]),
                                       np.log(fit_data[:, 1]))[0]
            add_rates[np.round(cur_rate, 2)] = fit_data
        elif rate is not None:
            ind = np.nonzero(np.array(direction) != 0)[0]
            if np.all(ind < len(rate)):
                add_rates[np.round(np.sum(rate[ind]), 2)] = line_data

    def _getLevelRate(rate, beta=None):
        if beta is not None:
            func = lambda x, r=rate, b=beta: b ** (r*x)
            label = r'${:.2g}^{{ {}\ell }}$'.format(beta,
                                                    _formatPower(rate))
        else:
            if 'x_axis' not in cur_kwargs or cur_kwargs['x_axis'] == 'ell':
                func = lambda x, r=rate: np.exp(r*x)
                label = r'$\exp({}\ell)$'.format(_formatPower(rate))
            else:
                func = lambda x, r=rate: x**r
                label = r'$x^{{ {} }}$'.format(_formatPower(rate))
        return func, label

    for j, r in enumerate(sorted(add_rates.keys(), key=lambda x:
                                 np.abs(x))):
        func, label = _getLevelRate(r, beta)
        ax.add_line(FunctionLine2D(fn=func, data=add_rates[r],
                                   linestyle=next(linestyles),
                                   c='k', label=label_fmt.format(label=label)))

@public
def plotErrorsVsTOL(ax, runs, *args, **kwargs):
    """Plots Errors vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes

    runs is a list
    run_data[i] is another class
    run_data[i].db_data.TOL
    run_data[i].db_data.finalTOL
    run_data[i].db_data.Creation_date
    run_data[i].db_data.niteration_index
    run_data[i].db_data.total_iterations
    run_data[i].db_data.total_time
    run_data[i] is an instance of mimc.MIMCRun
    run_data[i].data is an instance of mimc.MIMCData
    """

    num_kwargs = kwargs.pop('num_kwargs', None)
    modifier = kwargs.pop('modifier', None)
    relative = modifier is not None
    modifier = modifier if relative else 1.
    filteritr = kwargs.pop("filteritr", filteritr_convergent)

    ax.set_xlabel(r'\tol')
    ax.set_ylabel('Relative error' if relative else 'Error')
    ax.set_yscale('log')
    ax.set_xscale('log')

    xy = np.array([[itr.TOL,
                    itr.exact_error,
                    itr.total_error_est] for _, itr in enum_iter(runs, filteritr)])
    xy[:, 1:3] = xy[:, 1:3] * modifier

    TOLs, error_est = __get_stats(xy, staton=2)
    plotObj = []

    ErrEst_kwargs = kwargs.pop('ErrEst_kwargs', None)
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    sel = np.logical_and(np.isfinite(xy[:, 1]), xy[:, 1] >=
                         np.finfo(float).eps)
    if np.sum(sel) == 0:
        plotObj.append(None)
    else:
        plotObj.append(_scatter(ax, xy[sel, 0], xy[sel, 1], *args, **kwargs))

    if ErrEst_kwargs is not None:
        plotObj.append(ax.errorbar(TOLs, error_est[:, 1],
                                   yerr=[error_est[:, 1]-error_est[:, 0],
                                         error_est[:, 2]-error_est[:, 1]],
                                   **ErrEst_kwargs))
    if Ref_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(rate=1,
                                                          const=modifier, **Ref_kwargs)))

    if num_kwargs is not None:
        shift = num_kwargs.pop("shift", 1.5)
        xy = xy[xy[:,0].argsort(), :]
        for TOL, itr in itertools.groupby(xy, key=lambda x: x[0]):
            curItr = np.array(list(itr))
            invalid = np.sum(curItr[:, 0]*modifier <= curItr[:, 1])
            #print("for ", TOL, ":", invalid, "out of", len(curItr), "are wrong")
            invalid = np.round(100.*invalid / len(curItr))
            if invalid == 0:
                continue
            plotObj.append(ax.text(TOL, shift*TOL*modifier,
                                   "{:g}".format(invalid),
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   **num_kwargs))

    return xy[sel, :2], plotObj

@public
def plotWorkVsLvlStats(ax, runs, *args, **kwargs):
    work_bins = kwargs.pop('work_bins', 50)
    xi = kwargs.pop('x_axis', 'work').lower()
    filteritr = kwargs.pop("filteritr", filteritr_all)

    if xi == 'work':
        x_label = "Avg. work"
    elif xi == 'time':
        x_label = "Avg. running time"
    elif xi == 'tol':
        x_label = "Avg. tolerance"
    else:              raise ValueError('x_axis')

    mymax = lambda A, n: [np.max(A[:, i]) for i in xrange(0, A.shape[1])] if len(A) > 0 else [None]*n
    def fnItrStats(run, i):
        itr = run.iters[i]
        try:
            prev = next(run.iters[j].lvls_count for j in
                        xrange(i-1, 0, -1) if filteritr(r, j))
        except:
            prev = 0

        if xi == 'work':
            itr_stats = itr.calcTotalWork()
        elif xi == 'time':
            itr_stats = run.iter_total_times()[i]
        elif xi == 'tol':
            itr_stats = run.TOL

        lvl_stats = mymax(np.array([[
            1+np.max(j) if len(j) > 0 else 0,
            np.max(data) if len(data) > 0 else 0,
            len(data)] for j, data in itr.lvls_sparse_itr(prev)]), 3)
        return [np.log(itr_stats), itr_stats] + lvl_stats

    ax.set_xlabel(x_label)
    ax.set_xscale('log')

    xy_binned = computeIterationStats(runs,work_bins=work_bins,
                                      filteritr=filteritr,
                                      fnItrStats=fnItrStats,
                                      arr_fnAgg=[np.mean, np.mean] +
                                      [np.max]*3)
    xy_binned = xy_binned[:, 1:]
    plotObj = []
    maxrefine_kwargs = kwargs.pop('maxrefine_kwargs', None)
    active_kwargs = kwargs.pop('active_kwargs', None)

    dim_args, dim_kwargs = normalize_fmt(args, kwargs)
    # Max dimensions
    if (np.max(xy_binned[:, 1]) - np.min(xy_binned[:, 1])) > 100:
        ax2 = ax.twinx()
        ax2.set_yscale('log')
    else:
        ax2 = ax
    plotObj.append(ax2.plot(xy_binned[:, 0], xy_binned[:, 1],
                            *dim_args, **dim_kwargs))

    # Max level in all dimension
    if maxrefine_kwargs is not None:
        maxrefine_args, maxrefine_kwargs = normalize_fmt((), maxrefine_kwargs)
        plotObj.append(ax.plot(xy_binned[:, 0], xy_binned[:, 2],
                               *maxrefine_args, **maxrefine_kwargs))

    # Max active dim
    if active_kwargs is not None:
        active_args, active_kwargs = normalize_fmt((), active_kwargs)
        plotObj.append(ax.plot(xy_binned[:, 0], xy_binned[:, 3],
                               *active_args, **active_kwargs))

    ax.grid(True)
    return xy_binned[:, 0:2], plotObj

@public
def plotWorkVsMaxError(ax, runs, *args, **kwargs):
    modifier = kwargs.pop('modifier', None)
    relative = modifier is not None
    modifier = modifier if relative else 1.
    iter_stats_args = kwargs.pop('iter_stats_args', dict())
    fnWork = kwargs.pop('fnWork', lambda itr: itr.calcTotalWork())
    fnAggError = kwargs.pop("fnAggError", np.max)
    flip = kwargs.pop('flip', False)

    if flip:
        ax.set_ylabel('Work')
        ax.set_xlabel('Max Relative Error' if relative else 'Max Error')
    else:
        ax.set_xlabel('Work')
        ax.set_ylabel('Max Relative Error' if relative else 'Max Error')
    ax.set_yscale('log')
    ax.set_xscale('log')

    def fnItrStats(run, i, in_fn=fnWork, in_flip=flip, in_mod=modifier):
        itr = run.iters[i]
        work = in_fn(run, i)
        if in_flip:
            return [np.log(itr.exact_error), itr.exact_error, work,
                    in_mod*itr.total_error_est]
        else:
            return [np.log(work), work, itr.exact_error,
                    in_mod*itr.total_error_est]

    xy_binned = computeIterationStats(runs,
                                      fnItrStats=fnItrStats,
                                      arr_fnAgg=[np.mean, np.mean,
                                                 fnAggError, np.min],
                                      **iter_stats_args)
    if len(xy_binned) == 0:
        return None, []
    xy_binned = xy_binned[:, 1:]
    plotObj = []
    ErrEst_kwargs = kwargs.pop('ErrEst_kwargs', None)
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    Ref_ErrEst_kwargs = kwargs.pop('Ref_ErrEst_kwargs', Ref_kwargs)
    sel = np.nonzero(np.logical_and(np.isfinite(xy_binned[:, 1]), xy_binned[:, 1] >=
                                    np.finfo(float).eps))[0]
    if len(sel) == 0:
        plotObj.append(None)
    else:
        args, kwargs = normalize_fmt(args, kwargs)
        plotObj.append(ax.plot(xy_binned[:, 0], xy_binned[:, 1], *args, **kwargs))

    if ErrEst_kwargs is not None:
        ErrEst_args, ErrEst_kwargs = normalize_fmt((), ErrEst_kwargs)
        plotObj.append(ax.plot(xy_binned[:, 0], xy_binned[:, 2], *ErrEst_args, **ErrEst_kwargs))
    #sel = sel[np.log(xy_binned[sel, 0]) > np.median(np.log(xy_binned[:, 0]))]
    if len(sel) > 0 and Ref_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(data=xy_binned[sel[-4:], :2],
                                                          **Ref_kwargs)))
        if ErrEst_kwargs is not None:
            plotObj.append(ax.add_line(FunctionLine2D.ExpLine(data=xy_binned[sel, :][:, [0,2]],
                                                              **Ref_ErrEst_kwargs)))

    return xy_binned[:, :2], plotObj


@public
def plotAvgWorkVsTime(ax, runs, *args, **kwargs):
    work_bins = kwargs.pop('work_bins', 50)
    filteritr = kwargs.pop("filteritr", filteritr_all)
    ax.set_xlabel('Avg. work')
    ax.set_ylabel('Running time')
    ax.set_yscale('log')
    ax.set_xscale('log')

    def fnItrStats(run, i):
        itr = run.iters[i]
        return [itr.calcTotalWork()] + [itr.total_time]*3

    xy_binned = computeIterationStats(runs,
                                      work_bins=len(runs[0].iters),
                                      filteritr=filteritr,
                                      fnItrStats=fnItrStats,
                                      arr_fnAgg=[np.mean, np.mean, np.max, np.min])
    plotObj = []
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    sel = np.nonzero(np.logical_and(np.isfinite(xy_binned[:, 1]), xy_binned[:, 1] >=
                                    np.finfo(float).eps))[0]
    if len(sel) == 0:
        plotObj.append(None)
    else:
        plotObj.append(ax.errorbar(xy_binned[:, 0], xy_binned[:, 1],
                                   yerr=[xy_binned[:, 1]-xy_binned[:, 2],
                                         xy_binned[:, 3]-xy_binned[:, 1]],
                                   *args, **kwargs))
    if Ref_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(rate=1,
                                                          data=xy_binned[:, :2],
                                                          **Ref_kwargs)))

    return xy_binned[:, :2], plotObj

@public
def plotTotalWorkVsLvls(ax, runs, *args, **kwargs):
    """Plots Time vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Total Work')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    seed = kwargs.pop('seed', None)
    direction = kwargs.pop('direction', None)
    filteritr = kwargs.pop("filteritr", filteritr_all)

    if seed is None and direction is None:
        max_dim = kwargs.pop('max_dim')
    else:
        max_dim = len(seed) if seed is not None else len(direction)

    seed = np.array(seed) if seed is not None else np.zeros(max_dim, dtype=np.uint32)
    direction = np.array(direction) if direction is not None else np.ones(max_dim, dtype=np.uint32)

    plotObj = []
    iters = sorted(enum_iter(runs, filteritr), key=lambda itr: itr[1].TOL)
    label_fmt = kwargs.pop('label_fmt', None)
    TOLs = np.unique([itr.TOL for r, itr in iters])
    TOLs = TOLs[:kwargs.pop("max_TOLs", len(TOLs))]
    for TOL, iterator in itertools.groupby(iters, key=lambda itr: itr[1].TOL):
        if TOL not in TOLs:
            continue

        data_tw = []
        for r, curIter in iterator:
            cur = seed
            inds = []
            while True:
                ii = curIter.lvls_find(cur)
                if ii is None:
                    break
                inds.append(ii)
                cur = cur + direction
            for j, ind in enumerate(inds):
                data_tw.append([ind, curIter.tW[ind]])
        lvls, total_work = __get_stats(data_tw)
        plotObj.append(ax.errorbar(lvls, total_work[:, 1],
                                   yerr=[total_work[:, 1]-total_work[:, 0],
                                         total_work[:, 2]-total_work[:, 1]],
                                   label=label_fmt.format(TOL=TOL) if label_fmt is not None else None,
                                   *args,
                                   **kwargs))

    # TODO: Gotta figure out what this plot is about!!
    if curRun.params.min_dim == 1 and hasattr(curRun.params, "s") and hasattr(curRun.params, "gamma"):
        rate = np.array(curRun.params.gamma) - np.array(curRun.params.s)
        if hasattr(curRun.params, "beta"):
            rate *= np.log(curRun.params.beta)
        ax.add_line(FunctionLine2D(lambda x, tol=TOL, r=rate: tol**-2 * np.exp(r*x),
                                   data=np.array([lvls, total_work[:, 1]]).transpose(),
                                   linestyle='--', c='k'))
    return plotObj

def _get_x_axis(ax, ret, kwargs):
    x_axis = kwargs.pop("x_axis", "ell").lower()
    if x_axis == 'ell':
        ax.set_xlabel(r'$\ell$')
        x = np.arange(0, len(ret.central_delta_moments[:, 0]))
    elif x_axis == 'log_ell':
        ax.set_xlabel(r'$\ell$')
        x = np.log(1+np.arange(0, len(ret.central_delta_moments[:, 0])))
    elif x_axis == 'work':
        ax.set_xlabel(r'$W_\ell$')
        ax.set_xscale('log')
        x = np.mean(ret.Wl, axis=1)  # TODO: This should not take the average
    elif x_axis == 'time':
        ax.set_xlabel(r'$T_\ell$')
        ax.set_xscale('log')
        x = np.mean(ret.Tl, axis=1)  # TODO: This should not take the average
    else:
        raise ValueError('x_axis must be: ell, work or time not %s' % x_axis)
    return x, np.argsort(x)

@public
def plotExpectVsLvls(ax, runs, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    ax.set_ylabel(r'$E_\ell$')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fnNorm = kwargs.pop("fnNorm", np.abs)

    if "__calc_moments" in kwargs:
        ret = kwargs.pop("__calc_moments")
    else:
        ret = __calc_moments(runs,
                             seed=kwargs.pop('seed', None),
                             direction=kwargs.pop('direction', None),
                             fnNorm=fnNorm)

    fine_kwargs = kwargs.pop('fine_kwargs', None)
    plotObj = []
    El = ret.central_delta_moments[:, 0]
    x, sorted_ind = _get_x_axis(ax, ret, kwargs)

    if ret.central_delta_moments.shape[1] > 1:
        Vl = ret.central_delta_moments[:, 1]
        plotObj.append(ax.errorbar(x[sorted_ind],
                                   np.abs(El[sorted_ind]), *args,
                                   yerr=3*np.sqrt(np.abs(Vl[sorted_ind]/ret.M[sorted_ind])),
                                   **kwargs))
    else:
        plotObj.append(ax.plot(x[sorted_ind], np.abs(El[sorted_ind]),
                               *args, **kwargs))


    if fine_kwargs is not None:
        El = ret.central_fine_moments[:, 0]
        if ret.central_fine_moments.shape[1] > 1:
            Vl = ret.central_fine_moments[:, 1]
            plotObj.append(ax.errorbar(x[sorted_ind],
                                       np.abs(El[sorted_ind]),
                                       yerr=3*np.sqrt(np.abs(Vl[sorted_ind]/ret.M[sorted_ind])),
                                       **fine_kwargs))
        else:
            plotObj.append(ax.plot(x[sorted_ind], np.abs(El[sorted_ind]), **fine_kwargs))

    return plotObj[0][0].get_xydata(), plotObj

@public
def plotVarVsLvls(ax, runs, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    ax.set_ylabel(r'$V_\ell$')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm", np.abs)
    if "__calc_moments" in kwargs:
        ret = kwargs.pop("__calc_moments")
    else:
        ret = __calc_moments(runs,
                             seed=kwargs.pop('seed', None),
                             direction=kwargs.pop('direction',
                                                  None),
                             fnNorm=fnNorm)

    fine_kwargs = kwargs.pop('fine_kwargs', None)
    estimate_kwargs = kwargs.pop('estimate_kwargs', None)
    plotObj = []
    Vl = ret.central_delta_moments[:, 1]
    x, sorted_ind = _get_x_axis(ax, ret, kwargs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ret.central_delta_moments.shape[-1] >= 4:
        El4 = ret.central_delta_moments[:, 3]
        plotObj.append(ax.errorbar(x[sorted_ind], Vl[sorted_ind],
                                   yerr=3*np.sqrt(np.abs(El4[sorted_ind]/ret.M[sorted_ind])),
                                   *args, **kwargs))
    else:
        args, kwargs = normalize_fmt(args, kwargs)
        plotObj.append(ax.plot(x[sorted_ind], Vl[sorted_ind], *args, **kwargs))

    if fine_kwargs is not None:
        Vl = ret.central_fine_moments[:, 1]
        if ret.central_fine_moments.shape[-1] >= 4:
            El4 = ret.central_fine_moments[:, 3]
            plotObj.append(ax.errorbar(x[sorted_ind],
                                       Vl[sorted_ind],
                                       yerr=3*np.sqrt(np.abs(El4[sorted_ind]/ret.M[sorted_ind])),
                                       **fine_kwargs))
        else:
            fine_args, fine_kwargs = normalize_fmt((), fine_kwargs)
            plotObj.append(ax.plot(x[sorted_ind], Vl[sorted_ind],
                                   *fine_args, **fine_kwargs))

    if estimate_kwargs is not None:
        # mdat = np.ma.masked_array(ret.Vl_estimate, np.isnan(Vl_estimate))
        # med = np.ma.median(mdat, 1).filled(np.nan)
        min_vl = np.nanpercentile(ret.Vl_estimate, 5, axis=1)
        med = np.nanpercentile(ret.Vl_estimate, 50, axis=1)
        max_vl = np.nanpercentile(ret.Vl_estimate, 95, axis=1)
        #err = np.sqrt(((np.sum(ret.Vl_estimate**2, 1)/ret.M) - avg**2)/ret.M)
        plotObj.append(ax.errorbar(x[sorted_ind], med[sorted_ind],
                                   yerr=[med[sorted_ind]-min_vl[sorted_ind],
                                         max_vl[sorted_ind]-med[sorted_ind]],
                                   **estimate_kwargs))
    return plotObj[0][0].get_xydata(), plotObj

@public
def plotKurtosisVsLvls(ax, runs, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    args, kwargs = normalize_fmt(args, kwargs)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\textnormal{Kurt}_\ell$')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm", np.abs)
    if "__calc_moments" in kwargs:
        ret = kwargs.pop("__calc_moments")
    else:
        ret = __calc_moments(runs, seed=kwargs.pop('seed', None),
                             direction=kwargs.pop('direction',
                                                  None),
                             fnNorm=fnNorm)
    x, sorted_ind = _get_x_axis(ax, ret, kwargs)
    Vl = ret.central_delta_moments[:, 1]
    E4l = ret.central_delta_moments[:, 3]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    line = ax.plot(x[sorted_ind], E4l[sorted_ind]/Vl[sorted_ind]**2, *args, **kwargs)
    return line[0].get_xydata(), [line]

@public
def plotSkewnessVsLvls(ax, runs, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    args, kwargs = normalize_fmt(args, kwargs)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\textnormal{Skew}_\ell$')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm", np.abs)
    if "__calc_moments" in kwargs:
        ret = kwargs.pop("__calc_moments")
    else:
        ret = __calc_moments(runs, seed=kwargs.pop('seed', None),
                             direction=kwargs.pop('direction',
                                                  None),
                             fnNorm=fnNorm)
    x, sorted_ind = _get_x_axis(ax, ret, kwargs)
    Vl = ret.central_delta_moments[:, 1]
    E3l = np.abs(ret.central_delta_moments[:, 2])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    line = ax.plot(x[sorted_ind], E3l[sorted_ind]/Vl[sorted_ind]**1.5, *args, **kwargs)
    return line[0].get_xydata(), [line]

@public
def plotTimeVsLvls(ax, runs, *args, **kwargs):
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Time, [s]')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm", np.abs)
    if "__calc_moments" in kwargs:
        ret = kwargs.pop("__calc_moments")
    else:
        ret = __calc_moments(runs,
                             seed=kwargs.pop('seed', None),
                             direction=kwargs.pop('direction', None), fnNorm=fnNorm)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x, sorted_ind = _get_x_axis(ax, ret, kwargs)
    min_tl = np.nanpercentile(ret.Tl, 5, axis=1)
    med    = np.nanpercentile(ret.Tl, 50, axis=1)
    max_tl = np.nanpercentile(ret.Tl, 95, axis=1)
    line = ax.errorbar(x[sorted_ind], med[sorted_ind],
                       yerr=[med[sorted_ind]-min_tl[sorted_ind],
                             max_tl[sorted_ind]-med[sorted_ind]], *args, **kwargs)
    return line[0].get_xydata(), [line]

@public
def plotWorkVsLvls(ax, runs, *args, **kwargs):
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Work')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm", np.abs)
    if "__calc_moments" in kwargs:
        ret = kwargs.pop("__calc_moments")
    else:
        ret = __calc_moments(runs,
                             seed=kwargs.pop('seed', None),
                             direction=kwargs.pop('direction', None), fnNorm=fnNorm)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x, sorted_ind = _get_x_axis(ax, ret, kwargs)
    min_wl = np.nanpercentile(ret.Wl, 5, axis=1)
    med    = np.nanpercentile(ret.Wl, 50, axis=1)
    max_wl = np.nanpercentile(ret.Wl, 95, axis=1)
    line = ax.errorbar(x[sorted_ind], med[sorted_ind],
                       yerr=[med[sorted_ind]-min_wl[sorted_ind],
                             max_wl[sorted_ind]-med[sorted_ind]], *args,
                       **kwargs)
    return line[0].get_xydata(), [line]

@public
def plotTimeVsTOL(ax, runs, *args, **kwargs):
    """Plots Tl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    filteritr = kwargs.pop("filteritr", filteritr_all)
    fnTime = kwargs.pop("fnTime", 'time')
    MC_kwargs = kwargs.pop("MC_kwargs", None)
    min_samples = kwargs.pop("min_samples", None)
    scale_tol2 = kwargs.pop('scale_tol2', True)

    if scale_tol2:
        tol2_scale = lambda TOL: TOL**2
        tol2_label = r" $\times \tol^2$"
    else:
        tol2_scale = lambda TOL: 1.
        tol2_label = r""

    if fnTime == "work":
        ax.set_ylabel('Work estimate' + tol2_label)
        fnTime = lambda r, itr: np.sum(itr.tW * scalar(itr))
        fnMCWork = lambda r, itr: np.max(itr.calcWl())
    elif fnTime == "real_time":
        assert min_samples is None
        assert MC_kwargs is None
        ax.set_ylabel('Wall clock time [s]' + tol2_label)
        fnMCWork = lambda r, itr: 0.
        fnTime = lambda r, itr: r.iter_total_times[i]
    elif fnTime is 'time':
        ax.set_ylabel('Running time [s]' + tol2_label)
        fnTime = lambda r, itr: np.sum(itr.tT * scalar(itr))
        fnMCWork = lambda r, itr: np.max(itr.calcTl())
    elif fnTime == "max_work":
        assert min_samples is None
        ax.set_ylabel('Work estimate' + tol2_label)
        fnMCWork = lambda r, itr: np.max(itr.calcWl())
        fnTime = fnMCWork
    elif fnTime == "max_time":
        assert min_samples is None
        ax.set_ylabel('Running time [s]' + tol2_label)
        fnMCWork = lambda r, itr: np.max(itr.calcTl())
        fnTime = fnMCWork

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'\tol')

    scalar = lambda itr: 1
    if min_samples is not None:
        new_samples = lambda itr: np.maximum(np.float(min_samples),
                                             mimc._calcTheoryM(
                                                 itr.TOL,
                                                 active_lvls=itr.active_lvls,
                                                 theta=itr.parent._calcTheta(itr.TOL,
                                                                             itr.bias),
                                                 Vl=itr.parent.fn.Norm(itr.calcVl()),
                                                 Wl=itr.calcWl(),
                                                 ceil=False, Ca=itr.parent._Ca))
        if min_samples == 0:
            scalar = lambda itr: new_samples(itr) / itr.M
        else:
            scalar = lambda itr: np.ceil(new_samples(itr)) / itr.M

    if scale_tol2:
        tol2_scale = lambda TOL: TOL**2
    else:
        tol2_scale = lambda TOL: 1.
    xy = [[itr.TOL, fnTime(r, itr) * tol2_scale(itr.TOL),
           fnMCWork(r, itr) * r.estimateMonteCarloSampleCount(itr.TOL)
           * tol2_scale(itr.TOL)]
          for i, r, itr in enum_iter_i(runs, filteritr)]

    plotObj = []
    TOLs, times = __get_stats(xy)
    plotObj.append(ax.errorbar(TOLs, times[:, 1], *args,
                               yerr=[times[:, 1]-times[:, 0],
                                     times[:, 2]-times[:, 1]],
                               **kwargs))
    if MC_kwargs is not None:
        TOLs, times = __get_stats(xy, staton=2)
        plotObj.append(ax.errorbar(TOLs, times[:, 1], *args,
                                   yerr=[times[:, 1]-times[:, 0],
                                         times[:, 2]-times[:, 1]],
                                   **MC_kwargs))
    return plotObj[0][0].get_xydata(), plotObj

@public
def plotLvlsNumVsTOL(ax, runs, *args, **kwargs):
    """Plots L vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    lvl_index = kwargs.pop('lvl_index', None)
    ax.set_xscale('log')
    ax.set_xlabel(r'\tol')
    ax.set_ylabel(r'$L$')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    filteritr = kwargs.pop("filteritr", filteritr_all)
    summary = []
    for r in runs:
        prev = 0
        prevMax = 0
        for i in xrange(0, len(r.iters)):
            if not filteritr(r, i):
                continue
            itr = r.iters[i]
            if lvl_index is None:
                stats = [np.sum(data) for _, data in itr.lvls_sparse_itr(prev)]
            else:
                stats = [np.sum(data[lvl_index == j])
                         for j, data in itr.lvls_sparse_itr(prev)
                         if len(data[lvl_index == j]) > 0]
            if len(stats) == 0:
                assert(prev > 0)
                newMax = prevMax
            else:
                newMax = np.maximum(np.max(stats), prevMax)
            summary.append([itr.TOL, newMax])
            prev = itr.lvls_count
            prevMax = newMax

    summary = np.array(summary)

    a = summary
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    unique_a = a[idx]
    summary = a

    scatter = _scatter(ax, summary[:, 0], summary[:, 1], *args, **kwargs)
    return summary, [scatter]

@public
def plotThetaVsTOL(ax, runs, *args, **kwargs):
    """Plots theta vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    ax.set_xscale('log')
    ax.set_xlabel(r'\tol')
    ax.set_ylabel(r'$\theta$')
    ax.set_ylim([0, 1.])

    filteritr = kwargs.pop("filteritr", filteritr_all)
    summary = np.array([[itr.TOL, itr.Q.theta]
                        for _, itr in enum_iter(runs, filteritr)])

    scatter = _scatter(ax, summary[:, 0], summary[:, 1], *args, **kwargs)
    return summary, [scatter]

@public
def plotThetaRefVsTOL(ax, runs, eta, chi, *args, **kwargs):
    """Plots theta vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    ax.set_xscale('log')
    ax.set_xlabel(r'\tol')
    ax.set_ylabel(r'$\theta$')
    ax.set_ylim([0, 1.])

    filteritr = kwargs.pop("filteritr", filteritr_all)
    L = lambda itr: np.max([np.sum(l) for l in itr.lvls_itr()])
    if chi == 1:
        summary = np.array([[itr.TOL, (1. + (1./(2.*eta))*1./(L(itr)+1.))**-1]
                            for _, itr in enum_iter(runs, filteritr)])
    else:
        summary = np.array([[itr.TOL,
                             (1. + (1./(2.*eta))*(1.-chi)/(1.-chi**(L(itr)+1.)))**-1]
                            for _, itr in enum_iter(runs, filteritr)])
    TOL, thetaRef = __get_stats(summary, staton=1)

    #plotObj.append(ax.add_line(FunctionLine2D(lambda x: 1, *args, **kwargs)))
    line = ax.errorbar(TOL, thetaRef[:, 1],
                       yerr=[thetaRef[:, 1]-thetaRef[:, 0],
                             thetaRef[:, 2]-thetaRef[:, 1]],
                       *args, **kwargs)
    return summary, [line]

@public
def plotErrorsPP(ax, runs, label_fmt='${TOL}$', *args, **kwargs):
    """Plots Normal vs Empirical CDF of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    # Use TOL instead of finalTOL. The normality is proven w.r.t. to TOL of MLMC
    # not the finalTOL of MLMC (might be different sometimes)
    ax.set_xlabel(r'Empirical CDF')
    ax.set_ylabel("Normal CDF")
    ax.set_xlim([0, 1.])
    ax.set_ylim([0, 1.])

    smooth = kwargs.pop("smooth", False)
    filteritr = kwargs.pop("filteritr", filteritr_convergent)
    if "tol" not in kwargs:
        TOLs = [itr.TOL for _, itr in enum_iter(runs, filteritr)]
        unTOLs = np.unique(TOLs)
        unTOLs.sort()
        tol = unTOLs[np.argmax(np.bincount(np.digitize(TOLs, unTOLs)))-1]
    else:
        tol = kwargs.pop("tol")

    x_data = np.array([itr.calcEg() for _, itr in enum_iter(runs, filteritr) if
                       itr.TOL == tol])
    x = x_data - np.mean(x_data, axis=0)
    try:
        x = x.astype(np.float)
        if len(x.shape) != 1:
            raise Exception("Not correct size")
    except:
        plot_failed(ax)
        warnings.warn("PP plots require the object to implement __float__")
        return

    x /= np.std(x)
    ec = ECDF(x)
    plotObj = []
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    if label_fmt is not None:
        if 'label' in kwargs:
            raise ValueError('Cannot specify both label and label_fmt')
        kwargs['label'] = label_fmt.format(TOL=_format_latex_sci(tol))

    if smooth:
        xx = norm.cdf(x)
        arg = np.argsort(xx)
        plotObj.append(ax.plot(xx[arg], ec(x)[arg], *args, **kwargs)[0])
    else:
        plotObj.append(_scatter(ax, norm.cdf(x), ec(x), *args, **kwargs))

    if Ref_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(rate=1, **Ref_kwargs)))
    return (plotObj[0].get_offsets() if not smooth else plotObj[0].get_xydata()), plotObj

@public
def add_legend(ax, handles=None, labels=None, alpha=0.5, outside=None,
               loc='best', sort=False, *args, **kwargs):
    if not handles:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return None

    lines = np.array([type(h) is FunctionLine2D for h in handles], dtype=np.bool)

    # Order handles so that FunctionLine2D is always at the end
    if np.any(lines) and not np.all(lines):
        nlines = np.logical_not(lines)
        handles = np.array(handles)
        labels = np.array(labels)
        if sort:
            handles_lines, labels_lines = zip(*sorted(zip(handles[lines].tolist(), labels[lines].tolist()),
                                                      key=lambda t: t[1]))
            handles_nlines, labels_nlines = zip(*sorted(zip(handles[nlines].tolist(), labels[nlines].tolist()),
                                                        key=lambda t: t[1]))
        else:
            handles_lines, labels_lines = handles[lines].tolist(), labels[lines].tolist()
            handles_nlines, labels_nlines = handles[nlines].tolist(), labels[nlines].tolist()

        handles = handles_nlines + handles_lines
        labels = labels_nlines + labels_lines
    else:
        if sort:
            handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))

    if outside is not None and len(handles) >= outside:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        kwargs['loc'] = 'center left'
        kwargs['fancybox'] = False
        kwargs['frameon'] = False
        kwargs['shadow'] = False
        kwargs['bbox_to_anchor'] = (1., .5)
        return ax.legend(handles, labels, *args, **kwargs)
    else:
        kwargs['loc'] = loc
        kwargs['fancybox'] = False
        kwargs['frameon'] = False
        kwargs['shadow'] = False
        return ax.legend(handles, labels, *args, **kwargs)


def __formatMIMCRate(rate, log_rate, lbl_base=r"\tol",
                     scale_tol2=True, lbl_log_base=None):
    txt_rate = '{:.2g}'.format(rate)
    txt_log_rate = '{:.2g}'.format(log_rate)
    lbl_log_base = lbl_log_base or "{}^{{-1}}".format(lbl_base)
    label = lbl_base
    if txt_rate != "1":
        label += r'^{{ {} }}'.format(txt_rate)
    if txt_log_rate != "0":
        label += r'\log\left({}\right)^{{ {} }}'.format(lbl_log_base,
                                                        txt_log_rate)
    if scale_tol2:
        rate -= 2
    return (lambda x, r=rate, lr=log_rate: x**r * np.abs(np.log(x))**lr), \
        "${}$".format(label)


@public
def plot_failed(ax):
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax.text(0.5*(left+right), 0.5*(bottom+top), 'PLOTTING\nFAILED',
            horizontalalignment='center', verticalalignment='center',
            rotation=45, fontsize=60, color='red', alpha=0.5,
            transform=ax.transAxes)


@public
def plot_except(ax, verbose=True, reraise=True):
    plot_failed(ax)
    if verbose:
        print('-----------------------------------------------------')
        traceback.print_exc(limit=None)
        print('-----------------------------------------------------')
    if reraise:
        raise

@public
def genBooklet(runs, filteritr=None, input_args=dict(),
               modifier=1., fnNorm=None):
    if len(runs) == 1:
        raise Exception("Only one tag is supported")
    runs = runs[0]
    if len(runs) == 0:
        raise Exception("No runs!!!")

    filteritr = filteritr if filteritr is not None else filteritr_all
    reraise = False
    label_fmt = input_args.pop("label_fmt", None)
    call_add_legend = input_args.pop("add_legend", True)
    TOLs_count = len(np.unique([itr.TOL for _, itr in enum_iter(runs, filteritr)]))
    convergent_count = len([itr.TOL for _, itr in enum_iter(runs, filteritr_convergent)])
    iters_count = np.sum([len(r.iters) for r in runs])
    verbose = input_args.pop('verbose', False)
    legend_outside = input_args.pop("legend_outside", 5)
    if verbose:
        def print_msg(*args):
            print(*args)
    else:
        def print_msg(*args):
            return

    maxTOL = np.max([r.db_data.finalTOL for r in runs])
    params = next(r.params for r in runs if r.db_data.finalTOL == maxTOL)

    if fnNorm is None:
        fn = next(r.fn for r in runs if r.db_data.finalTOL == maxTOL)
        fnNorm = fn.Norm

    max_dim = np.max([np.max(r.last_itr.lvls_max_dim()) for r in runs])

    if label_fmt is None:
        label_fmt = '{label}'
        label_MIMC = "MIMC" if max_dim > 1 else "MLMC"
    else:
        label_MIMC = label_fmt.format(label='')

    any_bayesian = np.any([r.params.lsq_est for r in runs])
    has_gamma_rate = hasattr(params, 'gamma')
    has_w_rate = hasattr(params, 'w')
    has_s_rate = hasattr(params, 's')
    has_beta = hasattr(params, 'beta')

    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rc('font', **{'family': 'normal', 'weight': 'demibold',
                      'size': 15})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \def{\tol}{\ensuremath{\varepsilon}}')

    figures = []
    def add_fig():
        figures.append(plt.figure())
        return figures[-1].gca()

    if (TOLs_count > 1):
        print_msg("plotErrorsVsTOL")
        ax = add_fig()
        try:
            plotErrorsVsTOL(ax, runs, filteritr=filteritr,
                            modifier=modifier,
                            ErrEst_kwargs={'label':
                                           label_fmt.format(label='Error Estimate')},
                            Ref_kwargs={'ls': '--', 'c':'k',
                                        'label': label_fmt.format(label=r'\tol')},
                            num_kwargs={'color': 'r'})
        except:
            plot_except(ax, verbose, reraise)

    print_msg("plotWorkVsMaxError")
    ax = add_fig()
    Ref_kwargs = {'ls': '--', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}
    ErrEst_kwargs = {'fmt': '--*','label': label_fmt.format(label='Error Estimate')}
    Ref_ErrEst_kwargs = {'ls': '-.', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}
    try:
        plotWorkVsMaxError(ax, runs,
                           iter_stats_args=dict(filteritr=filteritr),
                           fnWork=lambda run, i: run.iters[i].calcTotalWork(),
                           modifier=modifier, fmt='-*',
                           ErrEst_kwargs=ErrEst_kwargs,
                           Ref_kwargs=Ref_kwargs,
                           Ref_ErrEst_kwargs=Ref_ErrEst_kwargs)
        ax.set_xlabel('Avg. Iteration Work')
    except:
        plot_except(ax, verbose, reraise)

    print_msg("plotWorkVsMaxError")
    ax = add_fig()
    try:
        plotWorkVsMaxError(ax, runs,
                           iter_stats_args=dict(filteritr=filteritr),
                           fnWork=lambda run, i: run.iters[i].calcTotalTime(),
                           modifier=modifier, fmt='-*',
                           ErrEst_kwargs=ErrEst_kwargs,
                           Ref_kwargs=Ref_kwargs,
                           Ref_ErrEst_kwargs=Ref_ErrEst_kwargs)
        ax.set_xlabel('Avg. Iteration Time')
    except:
        plot_except(ax, verbose, reraise)

    print_msg("plotAvgWorkVsTime")
    ax = add_fig()
    try:
        plotAvgWorkVsTime(ax, runs,
                          filteritr=filteritr,
                          fmt='-*',
                          Ref_kwargs={'ls': '--', 'c':'k',
                                      'label': label_fmt.format(label='{rate:.2g}')})
    except:
        plot_except(ax, verbose, reraise)

    print_msg("plotWorkVsLvlStats")
    ax = add_fig()
    try:
        plotWorkVsLvlStats(ax, runs, filteritr=filteritr,
                           fmt='-ob', label=label_fmt.format(label='Total dim.'),
                           active_kwargs={'fmt': '-*g',
                                          'label': label_fmt.format(label='Max active dim.')},
                           maxrefine_kwargs={'fmt': '-sr',
                                             'label': label_fmt.format(label='Max refinement')})
    except:
        plot_except(ax, verbose, reraise)

    if (convergent_count > 10):   # Need at least 10 plots for this plot to be significant
        print_msg("plotErrorsPP")
        ax = add_fig()
        try:
            # This plot only makes sense for convergent plots
            plotErrorsPP(ax, runs, filteritr=filteritr_convergent,
                         Ref_kwargs={'ls': '--', 'c': 'k'})
        except:
            plot_except(ax, verbose, reraise)


    if (TOLs_count > 1):
        print_msg("plotTimeVsTOL")
        ax_time = add_fig()
        print_msg("plotTimeVsTOL")
        try:
            _, _ = plotTimeVsTOL(ax_time, runs, fmt='-.',
                                 label=label_fmt.format(label="Wall clock"),
                                 filteritr=filteritr, fnTime="real_time",
                                 MC_kwargs=None)
        except:
            plot_except(ax_time, verbose, reraise)

        try:
            data_time, _ = plotTimeVsTOL(ax_time, runs, label=label_MIMC,
                                         filteritr=filteritr,
                                         MC_kwargs=None if max_dim > 1
                                         else {"label": label_fmt.format(label="MC Estimate"),
                                               "fmt": "--r"})
        except:
            plot_except(ax_time, verbose, reraise)

        print_msg("plotTimeVsTOL")
        ax_est = add_fig()
        try:
            data_est, _ = plotTimeVsTOL(ax_est, runs, label=label_MIMC,
                                        filteritr=filteritr,
                                        fnTime="work",
                                        MC_kwargs= None if max_dim > 1
                                        else {"label": label_fmt.format(label="MC Estimate"),
                                              "fmt": "--r"})
            if has_s_rate and has_gamma_rate and has_w_rate:
                s = np.array(params.s)
                w = np.array(params.w)
                gamma = np.array(params.gamma)
                if has_beta:
                    s = s * np.log(params.beta)
                    w = w * np.log(params.beta)
                    gamma = gamma * np.log(params.beta)
                func, label = __formatMIMCRate(*mimc.calcMIMCRate(w, s, gamma))
                ax_time.add_line(FunctionLine2D(fn=func,
                                                data=data_time,
                                                linestyle='--', c='k',
                                                label=label_fmt.format(label=label)))
                ax_est.add_line(FunctionLine2D(fn=func,
                                               data=data_est,
                                               linestyle='--', c='k',
                                               label=label_fmt.format(label=label)))
        except:
            plot_except(ax_est, verbose, reraise)

    lvl_funcs = [[0, False, False, plotTimeVsLvls, np.array(params.gamma)
                  if has_gamma_rate else None],
                 [0, False, False, plotWorkVsLvls, None],
                 [1, True, False, plotExpectVsLvls, -np.array(params.w)
                  if has_w_rate else None],
                 [2, True, any_bayesian, plotVarVsLvls, -np.array(params.s)
                  if has_s_rate else None],
                 [3, False, False, plotSkewnessVsLvls, None],
                 [4, False, False, plotKurtosisVsLvls, None]]
    max_moment = runs[0].last_itr.psums_delta.shape[1]
    for min_moment, plotFine, plotEstimate, plotFunc, rate in lvl_funcs:
        if min_moment > max_moment:
            continue
        try:
            print_msg(plotFunc.__name__)
            ax = add_fig()
            plotDirections(ax, runs, plotFunc, fnNorm=fnNorm,
                           label_fmt=label_fmt, plot_fine=plotFine,
                           plot_estimate=plotEstimate,
                           beta=params.beta if has_beta else None,
                           rate=rate)
        except:
            plot_except(ax, verbose, reraise)

    if TOLs_count > 1:
        print_msg("plotLvlsNumVsTOL")
        ax = add_fig()
        try:
            line_data, _ = plotLvlsNumVsTOL(ax, runs, filteritr=filteritr)
            if has_beta and has_w_rate and has_gamma_rate:
                if has_beta:
                    rate = 1./np.min(np.array(params.w) * np.log(params.beta))
                else:
                    rate = 1./np.min(np.array(params.w))
                label = r'${}\log\left({\tol}^{{-1}}\right)$'.format(_formatPower(rate))
                ax.add_line(FunctionLine2D(fn=lambda x, r=rate: -rate*np.log(x),
                                           data=line_data,
                                           log_data=False,
                                           linestyle='--', c='k',
                                           label=label_fmt.format(label=label)))
        except:
            plot_except(ax, verbose, reraise)

        print_msg("plotThetaVsTOL")
        ax = add_fig()
        try:
            plotThetaVsTOL(ax, runs)
            if max_dim == 1 and has_s_rate and has_w_rate and has_gamma_rate:
                chi = params.s[0]/params.gamma[0]
                eta = params.w[0]/params.gamma[0]
                plotThetaRefVsTOL(ax, runs, filteritr=filteritr, eta=eta, chi=chi, fmt='--k')
        except:
            plot_except(ax, verbose, reraise)

    if call_add_legend:
        for fig in figures:
            for ax in fig.axes:
                add_legend(ax, outside=legend_outside)
    return figures

@public
def set_exact_errors(runs, fnExactErr, filteritr=filteritr_all):
    itrs = [itr for _, itr in enum_iter(runs, filteritr)]
    errs = fnExactErr(itrs)
    for i, itr in enumerate(itrs):
        itr.exact_error = errs[i]

@public
def estimate_exact(runs):
    minErr = np.min([r.total_error_est for r in runs])
    d = [r.calcEg() for r in runs if r.total_error_est == minErr]
    return np.sum(d, axis=0) * (1./ len(d)), minErr


def append_ext(path, ext):
    if path.lower().endswith(ext.lower()):
        return path
    return path + ext


def run_plot_program(fnPlot=genBooklet, fnExactErr=None, **kwargs):
    warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                             "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                       lineno, cat.__name__, msg)
    try:
        from matplotlib.cbook import MatplotlibDeprecationWarning
        warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
    except:
        pass   # Ignore

    def addExtraArguments(parser):
        parser.register('type', 'bool', lambda v: v.lower() in ("yes",
                                                                "true",
                                                                "t", "1"))
        parser.add_argument("-db_name", type=str, action="store",
                            help="Database Name")
        parser.add_argument("-db_engine", type=str, action="store",
                            help="Database Name")
        parser.add_argument("-db_user", type=str, action="store",
                            help="Database User")
        parser.add_argument("-db_host", type=str, action="store",
                            help="Database Host")
        parser.add_argument("-db_tag", type=str, action="store",
                            nargs='+', help="Database Tags")
        parser.add_argument("-label_fmt", type=str, action="store",
                            default=None, help="Output labels")
        parser.add_argument("-qoi_exact", type=float, action="store",
                            help="Exact value")
        parser.add_argument("-o", type=str,
                            action="store", help="Output file")
        parser.add_argument("-cmd", type=str, action="store",
                            help="Command to execute after plotting")
        parser.add_argument("-verbose", type='bool', action="store",
                            default=False)
        parser.add_argument("-filteritr", type=str, action="store",
                            choices=['convergent', 'all', 'last'])
        parser.add_argument("-relative", type='bool', action="store",
                            default=True)
        parser.add_argument("-done_flag", type=int, nargs='+',
                            action="store", default=None)
        parser.add_argument("-qoi_exact_tag", type=str, action="store")
        parser.add_argument("-formats", type=str, action="store",
                            nargs="+", default=["pdf"])

    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    args = test.parse_known_args(parser)
    for k in kwargs.keys():
        args.__dict__[k] = kwargs[k]

    db_args = dict()
    if args.db_name is not None:
        db_args["db"] = args.db_name
    if args.db_user is not None:
        db_args["user"] = args.db_user
    if args.db_host is not None:
        db_args["host"] = args.db_host
    if args.db_engine is not None:
        db_args["engine"] = args.db_engine
    if args.o is None:
        warnings.warn("Outputting to mimclib_out.pdf")
        args.o = "mimclib_out"

    db = mimcdb.MIMCDatabase(**db_args)
    if args.db_tag is None:
        warnings.warn("You did not select a database tag!!")
    if args.verbose:
        print("Reading data")

    total_runs = 0
    run_data = []
    for db_tag in args.db_tag:
        run_data.append(db.readRuns(tag=db_tag, done_flag=args.done_flag))
        total_runs += len(run_data[-1])

    if total_runs == 0:
        raise Exception("No runs!!!")

    fnNorm = run_data[0][0].fn.Norm
    if args.verbose:
        print("Plotting data")

    if args.qoi_exact_tag is not None:
        assert args.qoi_exact is None, "An exact value and exact tag are given"
        if args.qoi_exact_tag in args.db_tag:
            args.qoi_exact, _ = estimate_exact(run_data[args.db_tag.index(args.qoi_exact_tag)])
        else:
            exact_runs = db.readRuns(tag=args.qoi_exact_tag, done_flag=args.done_flag)
            args.qoi_exact, _ = estimate_exact(exact_runs)
        if args.verbose:
            print("Estimated exact value is {}".format(args.qoi_exact))

    if args.qoi_exact is not None:
        assert fnExactErr is None, "An exact value and an exact function are given"
        fnExactErr = lambda itrs, e=args.qoi_exact: \
                     fnNorm([v.calcEg() + e*-1 for v in itrs])
        modifier = 1./fnNorm([args.qoi_exact])
    else:
        # TODO: Need to somehow set it as relative
        modifier = 1.

    filteritr = None
    if args.filteritr == 'last':
        filteritr = filteritr_last
    elif args.filteritr == 'convergent':
        filteritr = filteritr_convergent
    elif args.filteritr == 'all':
        filteritr = filteritr_all

    if fnExactErr is not None:
        if args.verbose:
            print("Setting errors")
        for r in run_data:
            set_exact_errors(r, fnExactErr, filteritr=filteritr)

    figures = fnPlot(run_data,
                     modifier=modifier if args.relative else None,
                     fnNorm=fnNorm,
                     filteritr=filteritr,
                     input_args=vars(args).copy())

    if args.verbose:
        print("Saving file")

    if 'pdf' in args.formats:
        save_pdf(append_ext(args.o, '.pdf'), figures, close=False)

    if 'tikz' in args.formats:
        dir_name = args.o
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for i, fig in enumerate(figures):
            print("Saving", fig.label)
            tikz_save("{}/{}.tex".format(dir_name, fig.label), fig,
                      show_info=False,
                      figurewidth=r'\figurewidth',
                      figureheight=r'\figureheight',
                      figlabel=r'\figlabel')

    if args.cmd is not None:
        os.system(args.cmd.format(args.o))
