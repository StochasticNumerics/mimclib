from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
from . import mimc
from . import ipdb
from matplotlib.ticker import MaxNLocator

__all__ = []


def public(sym):
    __all__.append(sym.__name__)
    return sym

@public
class FunctionLine2D(plt.Line2D):
    def __init__(self, *args, **kwargs):
        self.flip = kwargs.pop('flip', False)
        self.fn = fn = kwargs.pop('fn')
        log_data = kwargs.pop('log_data', True)
        data = kwargs.pop('data', None)
        if data is not None:
            x = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            if len(x) > 0 and len(y) > 0:
                if log_data:
                    const = [np.mean(y/fn(x)), 0]
                    self.fn = lambda x, cc=const, ff=fn: cc[0] * ff(x) + cc[1]
                else:
                    const = [np.mean(y-fn(x)), 0]
                    self.fn = lambda x, cc=const, ff=fn: ff(x) + cc[0]

        super(FunctionLine2D, self).__init__([], [], *args, **kwargs)

    def _linspace(self, lim, scale, N=100):
        if scale == 'log':
            return np.exp(np.linspace(np.log(lim[0]), np.log(lim[1]), N))
        else:
            return np.linspace(lim[0], lim[1], N)

    def draw(self, renderer):
        import matplotlib.pylab as plt
        ax = self.get_axes()
        if self.flip:
            y = self._linspace(ax.get_ylim(), ax.get_yscale())
            self.set_xdata(self.fn(y))
            self.set_ydata(y)
        else:
            x = self._linspace(ax.get_xlim(), ax.get_xscale())
            self.set_xdata(x)
            self.set_ydata(self.fn(x))

        plt.Line2D.draw(self, renderer)
        self.set_xdata([])
        self.set_ydata([])

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
                    rate = np.polyfit(np.log(x), np.log(y), 1)[0]
                else:
                    rate = np.polyfit(x, y, 1)[0]
        if "label" in kwargs:
            kwargs["label"] = kwargs.pop("label").format(rate=rate)
        return FunctionLine2D(*args, fn=lambda x, r=rate:
                              const*np.array(x)**r, data=data,
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
    import itertools
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


def filteritr_last(run, iter_idx):
    return len(run.iters)-1 == iter_idx

def filteritr_convergent(run, iter_idx):
    return run.iters[iter_idx].totalErrorEst() <= run.iters[iter_idx].TOL

def filteritr_all(run, iter_idx):
    return True

def enum_iter(runs, fnFilter):
    for r in runs:
        for i in xrange(0, len(r.iters)):
            if fnFilter(r, i):
                yield r, r.iters[i]

def estimate_exact(runs):
    minErr = np.min([r.totalErrorEst() for r in runs])
    exact = np.mean([r.calcEg() for r in runs if
                     r.totalErrorEst() == minErr], axis=0)
    return exact

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
    run_data[i].db_data.totalTime
    run_data[i] is an instance of mimc.MIMCRun
    run_data[i].data is an instance of mimc.MIMCData
    """

    fnNorm = kwargs.pop('fnNorm')
    num_kwargs = kwargs.pop('num_kwargs', None)
    exact = kwargs.pop('exact')
    relative_error = kwargs.pop('relative', True)
    filteritr = kwargs.pop("filteritr", filteritr_all)

    modifier = (1./fnNorm(np.array([exact]))[0]) if relative_error else 1.
    val = np.array([itr.calcEg() for _, itr in enum_iter(runs, filteritr)])
    xy = np.array([[itr.TOL, 0, itr.totalErrorEst()] for _, itr in enum_iter(runs, filteritr)])
    xy[:, 1] = fnNorm(val-exact)
    xy[:, 1:3] = xy[:, 1:3] * modifier

    TOLs, error_est = __get_stats(xy, staton=2)
    plotObj = []

    ax.set_xlabel('TOL')
    ax.set_ylabel('Errors')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ErrEst_kwargs = kwargs.pop('ErrEst_kwargs')
    Ref_kwargs = kwargs.pop('Ref_kwargs')
    sel = np.logical_and(np.isfinite(xy[:, 1]), xy[:, 1] >=
                         np.finfo(float).eps)
    if np.sum(sel) == 0:
        plotObj.append(None)
    else:
        plotObj.append(ax.scatter(xy[sel, 0], xy[sel, 1], *args, **kwargs))

    if ErrEst_kwargs is not None:
        plotObj.append(ax.errorbar(TOLs, error_est[:, 1],
                                   yerr=[error_est[:, 1]-error_est[:, 0],
                                         error_est[:, 2]-error_est[:, 1]],
                                   **ErrEst_kwargs))
    if Ref_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(rate=1,
                                                          const=modifier, **Ref_kwargs)))

    if num_kwargs is not None:
        import itertools
        xy = xy[xy[:,0].argsort()]
        for TOL, itr in itertools.groupby(xy, key=lambda x: x[0]):
            curItr = np.array(list(itr))
            invalid = np.sum(curItr[:, 0]*modifier <= curItr[:, 1])
            if invalid == 0:
                continue
            ax.text(TOL, 1.5*TOL*modifier,
                    "{:g}\%".format(np.round(100.*invalid / len(curItr))),
                    horizontalalignment='center',
                    verticalalignment='center',
                    **num_kwargs)

    return xy[sel, :2], plotObj

def computeIterationStats(runs, work_bins, xi, filteritr, fnNorm=None,
                          exact=None, relative=False):
    if xi == 'work':
        xi = 0
        x_label = "Avg. work"
    elif xi == 'time':
        xi = 1
        x_label = "Avg. running time"
    elif xi == 'tol':
        xi = 2
        x_label = "Avg. tolerance"
    else:              raise ValueError('x_axis')

    if exact is None:
        modifier = 1
    else:
        modifier = (1./fnNorm(np.array([exact]))[0]) if relative else 1.
    val = np.array([itr.calcEg() for _, itr in enum_iter(runs, filteritr)])

    mymax = lambda A: [np.max(A[:, i]) for i in xrange(0, A.shape[1])]
    xy = []
    val = []
    prev = 0
    for _, itr in enum_iter(runs, filteritr):
        stats = mymax(np.array([[
            1+np.max(j) if len(j) > 0 else 0,
            np.max(data) if len(data) > 0 else 0,
            len(data)] for j, data in itr.lvls_sparse_itr(prev)]))

        if len(xy) > 0:
            stats = mymax(np.vstack((stats, xy[-1][5:])))
        xy.append([itr.calcTotalWork(), itr.calcTotalTime(),
                   itr.TOL, 0, modifier*itr.totalErrorEst()]+stats)
        val.append(itr.calcEg())
        prev = itr.lvls_count
    xy = np.array(xy)
    if exact is not None:
        xy[:, 3] = modifier*fnNorm(val-exact)

    lxy = np.log(xy[:, xi])
    bins = np.digitize(lxy, np.linspace(np.min(lxy), np.max(lxy), work_bins))
    bins[bins == work_bins] = work_bins-1
    ubins = np.unique(bins)
    xy_binned = np.zeros((len(ubins), 6))
    for i, b in enumerate(ubins):
        d = xy[bins==b, :]
        xy_binned[i, 0] = np.mean(d[:, xi])  # Mean work
        xy_binned[i, 1] = np.max(xy[bins==b, 3])  # Max exact error
        xy_binned[i, 2] = np.min(xy[bins==b, 4])  # min error estimate

        xy_binned[i, 3] = np.max(d[:, 5])  # Max dimension
        xy_binned[i, 4] = np.max(d[:, 6])  # max level in all dim
        xy_binned[i, 5] = np.max(d[:, 7])  # max active dim

    xy_binned = xy_binned[xy_binned[:,0].argsort(), :]
    return xy_binned

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

    xy_binned = computeIterationStats(runs, xi=xi,
                                      work_bins=work_bins,
                                      filteritr=filteritr)

    plotObj = []
    ax.set_xlabel(x_label)
    #ax.set_ylabel('??')
    ax.set_xscale('log')

    maxlvl_kwargs = kwargs.pop('maxlvl_kwargs', None)
    active_kwargs = kwargs.pop('active_kwargs', None)

    maxdim_args, maxdim_kwargs = __normalize_fmt(args, kwargs)
    # Max dimensions
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    plotObj.append(ax2.plot(xy_binned[:, 0], xy_binned[:, 3],
                            *maxdim_args, **maxdim_kwargs))

    # Max level in all dimension
    if maxlvl_kwargs is not None:
        maxlvl_args, maxlvl_kwargs = __normalize_fmt((), maxlvl_kwargs)
        plotObj.append(ax.plot(xy_binned[:, 0], xy_binned[:, 4],
                               *maxlvl_args, **maxlvl_kwargs))

    # Max active dim
    if maxlvl_kwargs is not None:
        active_args, active_kwargs = __normalize_fmt((), active_kwargs)
        plotObj.append(ax.plot(xy_binned[:, 0], xy_binned[:, 5],
                               *active_args, **active_kwargs))

    return xy_binned[:, [0,3]], plotObj


@public
def plotWorkVsMaxError(ax, runs, *args, **kwargs):
    fnNorm = kwargs.pop('fnNorm')
    exact = kwargs.pop('exact')
    relative = kwargs.pop('relative', True)
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

    xy_binned = computeIterationStats(runs, xi=xi,
                                      work_bins=work_bins,
                                      fnNorm=fnNorm,
                                      relative=relative,
                                      filteritr=filteritr,
                                      exact=exact)
    plotObj = []

    ax.set_xlabel(x_label)
    ax.set_ylabel('Max Relative Error' if relative else 'Max Error')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ErrEst_kwargs = kwargs.pop('ErrEst_kwargs')
    Ref_kwargs = kwargs.pop('Ref_kwargs')
    sel = np.logical_and(np.isfinite(xy_binned[:, 1]), xy_binned[:, 1] >=
                         np.finfo(float).eps)
    if np.sum(sel) == 0:
        plotObj.append(None)
    else:
        args, kwargs = __normalize_fmt(args, kwargs)
        plotObj.append(ax.plot(xy_binned[:, 0], xy_binned[:, 1], *args, **kwargs))

    if ErrEst_kwargs is not None:
        ErrEst_args, ErrEst_kwargs = __normalize_fmt((), ErrEst_kwargs)
        plotObj.append(ax.plot(xy_binned[:, 0], xy_binned[:, 2], *ErrEst_args, **ErrEst_kwargs))

    if Ref_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(data=xy_binned[sel, :2],
                                                          **Ref_kwargs)))
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(data=xy_binned[sel, :][:, [0,2]],
                                                          **Ref_kwargs)))

    return xy_binned[sel, :2], plotObj


def __calc_moments(runs, seed=None, direction=None, fnNorm=None):
    dim = len(seed) if seed is not None else len(direction)
    seed = np.array(seed) if seed is not None else np.zeros(dim, dtype=np.uint32)
    direction = np.array(direction) if direction is not None else np.ones(dim, dtype=np.uint32)
    moments = runs[0].last_itr.psums_delta.shape[1]
    psums_delta, psums_fine, Tl, Vl_estimate, M = [None]*5
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
            Vl_estimate[:, i] = curRun.Vl_estimate[inds]
            Tl[:, i] = curRun.last_itr.tT[inds]/curRun.last_itr.M[inds]
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
            Vl_estimate[tmp:] = np.nan

            tmp = Tl.shape[0]
            Tl.resize((L, len(runs)), refcheck=False)
            Tl[tmp:] = np.nan

        Vl_estimate[:L, i] = curRun.Vl_estimate[inds]
        Vl_estimate[(L+1):, i] = np.nan
        Tl[:L, i] = curRun.last_itr.tT[inds]/curRun.last_itr.M[inds]
        Tl[(L+1):, i] = np.nan

    central_delta_moments = np.empty((psums_delta.shape[0],
                                      psums_delta.shape[1]), dtype=float)
    central_fine_moments = np.empty((psums_fine.shape[0],
                                     psums_fine.shape[1]), dtype=float)
    for m in range(1, psums_delta.shape[1]+1):
        central_delta_moments[:, m-1] = fnNorm(mimc.compute_central_moment(psums_delta, M, m))
        central_fine_moments[:, m-1] = fnNorm(mimc.compute_central_moment(psums_fine, M, m))

    return central_delta_moments, central_fine_moments, Tl, M, Vl_estimate

def __normalize_fmt(args, kwargs):
    if "fmt" in kwargs:        # Normalize behavior of errorbar() and plot()
        args = (kwargs.pop('fmt'), ) + args
    return args, kwargs

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
    import itertools
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
                data_tw.append([ind, curIter.M[ind] * curIter.Wl_estimate[ind]])
        lvls, total_work = __get_stats(data_tw)
        plotObj.append(ax.errorbar(lvls, total_work[:, 1],
                                   yerr=[total_work[:, 1]-total_work[:, 0],
                                         total_work[:, 2]-total_work[:, 1]],
                                   label=label_fmt.format(TOL) if label_fmt is not None else None,
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



@public
def plotExpectVsLvls(ax, runs, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$E_\ell$')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fnNorm = kwargs.pop("fnNorm")
    if "__calc_moments" in kwargs:
        central_delta_moments, central_fine_moments, _, M, _ = kwargs.pop("__calc_moments")
    else:
        central_delta_moments, central_fine_moments, _, M, _ = __calc_moments(runs,
                                                                              seed=kwargs.pop('seed', None),
                                                                              direction=kwargs.pop('direction', None),
                                                                              fnNorm=fnNorm)

    fine_kwargs = kwargs.pop('fine_kwargs', None)
    plotObj = []
    El = central_delta_moments[:, 0]
    if central_delta_moments.shape[1] > 1:
        Vl = central_delta_moments[:, 1]
        plotObj.append(ax.errorbar(np.arange(0, len(El)), np.abs(El), *args,
                                   yerr=3*np.sqrt(np.abs(Vl/M)), **kwargs))
    else:
        plotObj.append(ax.plot(np.arange(0, len(El)), np.abs(El), *args, **kwargs))


    if fine_kwargs is not None:
        El = central_fine_moments[:, 0]
        if central_fine_moments.shape[1] > 1:
            Vl = central_fine_moments[:, 1]
            plotObj.append(ax.errorbar(np.arange(0, len(El)), np.abs(El),
                                       yerr=3*np.sqrt(np.abs(Vl/M)), **fine_kwargs))
        else:
            plotObj.append(ax.plot(np.arange(0, len(El)), np.abs(El), **fine_kwargs))

    return plotObj[0][0].get_xydata(), plotObj


@public
def plotVarVsLvls(ax, runs, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$V_\ell$')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm")
    if "__calc_moments" in kwargs:
        central_delta_moments, central_fine_moments, \
            _, M, Vl_estimate = kwargs.pop("__calc_moments")
    else:
        central_delta_moments, central_fine_moments, \
            _, M, Vl_estimate = __calc_moments(runs,
                                               seed=kwargs.pop('seed', None),
                                               direction=kwargs.pop('direction',
                                                                    None),
                                               fnNorm=fnNorm)
    fine_kwargs = kwargs.pop('fine_kwargs', None)
    estimate_kwargs = kwargs.pop('estimate_kwargs', None)
    plotObj = []
    Vl = central_delta_moments[:, 1]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if central_delta_moments.shape[-1] >= 4:
        El4 = central_delta_moments[:, 3]
        plotObj.append(ax.errorbar(np.arange(0, len(Vl)), Vl,
                                   yerr=3*np.sqrt(np.abs(El4/M)),
                                   *args, **kwargs))
    else:
        args, kwargs = __normalize_fmt(args, kwargs)
        plotObj.append(ax.plot(np.arange(0, len(Vl)), Vl, *args, **kwargs))

    if fine_kwargs is not None:
        Vl = central_fine_moments[:, 1]
        if central_fine_moments.shape[-1] >= 4:
            El4 = central_fine_moments[:, 3]
            plotObj.append(ax.errorbar(np.arange(0, len(Vl)), Vl,
                                   yerr=3*np.sqrt(np.abs(El4/M)),
                                   **fine_kwargs))
        else:
            fine_args, fine_kwargs = __normalize_fmt((), fine_kwargs)
            plotObj.append(ax.plot(np.arange(0, len(Vl)), Vl, *fine_args, **fine_kwargs))

    if estimate_kwargs is not None:
        # mdat = np.ma.masked_array(Vl_estimate, np.isnan(Vl_estimate))
        # med = np.ma.median(mdat, 1).filled(np.nan)
        min_vl = np.nanpercentile(Vl_estimate, 5, axis=1)
        med = np.nanpercentile(Vl_estimate, 50, axis=1)
        max_vl = np.nanpercentile(Vl_estimate, 95, axis=1)
        #err = np.sqrt(((np.sum(Vl_estimate**2, 1)/M) - avg**2)/M)
        plotObj.append(ax.errorbar(np.arange(0, len(Vl)),
                                   med, yerr=[med-min_vl, max_vl-med],
                                   **estimate_kwargs))
    return plotObj[0][0].get_xydata(), plotObj


@public
def plotKurtosisVsLvls(ax, runs, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    args, kwargs = __normalize_fmt(args, kwargs)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\textnormal{Kurt}_\ell$')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm")
    if "__calc_moments" in kwargs:
        central_delta_moments, _,  _, _, _ = kwargs.pop("__calc_moments")
    else:
        central_delta_moments, _, _, _, _ = __calc_moments(runs,
                                                        seed=kwargs.pop('seed', None),
                                                        direction=kwargs.pop('direction',
                                                                             None),
                                                           fnNorm=fnNorm)
    Vl = central_delta_moments[:, 1]
    E4l = central_delta_moments[:, 3]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    line = ax.plot(np.arange(0, len(Vl)), E4l/Vl**2, *args, **kwargs)
    return line[0].get_xydata(), [line]


@public
def plotSkewnessVsLvls(ax, runs, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    args, kwargs = __normalize_fmt(args, kwargs)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\textnormal{Skew}_\ell$')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm")
    if "__calc_moments" in kwargs:
        central_delta_moments, _, _, _, _ = kwargs.pop("__calc_moments")
    else:
        central_delta_moments, _, _, _, _ = __calc_moments(runs,
                                                           seed=kwargs.pop('seed', None),
                                                           direction=kwargs.pop('direction',
                                                                                None),
                                                           fnNorm=fnNorm)
    Vl = central_delta_moments[:, 1]
    E3l = np.abs(central_delta_moments[:, 2])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    line = ax.plot(np.arange(0, len(Vl)), E3l/Vl**1.5, *args, **kwargs)
    return line[0].get_xydata(), [line]



@public
def plotTimeVsLvls(ax, runs, *args, **kwargs):
    """Plots Time vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')
    fnNorm = kwargs.pop("fnNorm")
    if "__calc_moments" in kwargs:
        _, _, Tl, M, _ = kwargs.pop("__calc_moments")
    else:
        _, _, Tl, M, _ = __calc_moments(runs,
                                        seed=kwargs.pop('seed', None),
                                        direction=kwargs.pop('direction', None), fnNorm=fnNorm)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    min_tl = np.nanpercentile(Tl, 5, axis=1)
    med = np.nanmean(Tl, axis=1)
    max_tl = np.nanpercentile(Tl, 95, axis=1)
    line = ax.errorbar(np.arange(0, len(Tl)),
                       med, yerr=[med-min_tl, max_tl-med],
                       *args, **kwargs)
    return line[0].get_xydata(), [line]


@public
def plotTimeVsTOL(ax, runs, *args, **kwargs):
    """Plots Tl vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    filteritr = kwargs.pop("filteritr", filteritr_all)
    work_estimate = kwargs.pop("work_estimate", False)
    if kwargs.pop("real_time", False):
        if work_estimate:
            raise ValueError("real_time and work_estimate cannot be both True")
        if 'MC_kwargs' in kwargs:
            raise ValueError("Cannot estimate real time of Monte Carlo")

        xy = [[r.db_data.finalTOL, r.db_data.totalTime] for r in runs]
    elif work_estimate:
        xy = [[itr.TOL, np.sum(itr.M*itr.Wl_estimate),
               np.max(itr.Wl_estimate) * r.estimateMonteCarloSampleCount(itr.TOL)]
              for r, itr in enum_iter(runs, filteritr)]
    else:
        xy = [[itr.TOL, np.sum(itr.tT),
               np.max(itr.calcTl()) * r.estimateMonteCarloSampleCount(itr.TOL)]
              for r, itr in enum_iter(runs, filteritr)]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('TOL')
    if work_estimate:
        ax.set_ylabel('Work estimate')
    else:
        ax.set_ylabel('Average Time (s)')

    plotObj = []
    TOLs, times = __get_stats(xy)
    MC_kwargs = kwargs.pop("MC_kwargs", None)

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
    filteritr = kwargs.pop("filteritr", filteritr_all)
    summary = np.array([[itr.TOL,
                         np.max([np.sum(l) for l in itr.lvls_itr()])]
                        for _, itr in enum_iter(runs, filteritr)])

    ax.set_xscale('log')
    ax.set_xlabel('TOL')
    ax.set_ylabel(r'$L$')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    scatter = ax.scatter(summary[:, 0], summary[:, 1], *args, **kwargs)
    return summary, [scatter]


@public
def plotThetaVsTOL(ax, runs, *args, **kwargs):
    """Plots theta vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    filteritr = kwargs.pop("filteritr", filteritr_all)
    summary = np.array([[itr.TOL, itr.Q.theta]
                        for _, itr in enum_iter(runs, filteritr)])

    ax.set_xscale('log')
    ax.set_xlabel('TOL')
    ax.set_ylabel(r'$\theta$')
    ax.set_ylim([0, 1.])
    scatter = ax.scatter(summary[:, 0], summary[:, 1], *args, **kwargs)
    return summary, [scatter]


@public
def plotThetaRefVsTOL(ax, runs, eta, chi, *args, **kwargs):
    """Plots theta vs TOL of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    filteritr = kwargs.pop("filteritr", filteritr_all)
    L = lambda itr: np.max([np.sum(l) for l in itr.lvls_itr()])
    if chi == 1:
        summary = np.array([[itr.TOL, (1. + (1./(2.*eta))*1./(L(itr)+1.))**-1]
                            for _, itr in enum_iter(runs, filteritr)])
    else:
        summary = np.array([[itr.TOL,
                             (1. + (1./(2.*eta))*(1.-chi)/(1.-chi**(L(r)+1.)))**-1]
                            for _, itr in enum_iter(runs, filteritr)])
    TOL, thetaRef = __get_stats(summary, staton=1)

    #plotObj.append(ax.add_line(FunctionLine2D(lambda x: 1, *args, **kwargs)))
    ax.set_xscale('log')
    ax.set_xlabel('TOL')
    ax.set_ylabel(r'$\theta$')
    ax.set_ylim([0, 1.])
    line = ax.errorbar(TOL, thetaRef[:, 1],
                       yerr=[thetaRef[:, 1]-thetaRef[:, 0],
                             thetaRef[:, 2]-thetaRef[:, 1]],
                       *args, **kwargs)
    return summary, [line]

@public
def plotErrorsQQ(ax, runs, *args, **kwargs):
    """Plots Normal vs Empirical CDF of @runs, as
    returned by MIMCDatabase.readRunData()
    ax is in instance of matplotlib.axes
    """
    # Use TOL instead of finalTOL. The normality is proven w.r.t. to TOL of MLMC
    # not the finalTOL of MLMC (might be different sometimes)
    filteritr = kwargs.pop("filteritr", filteritr_convergent)

    if "tol" not in kwargs:
        TOLs = [itr.TOL for _, itr in enum_iter(runs, filteritr)]
        unTOLs = np.unique(TOLs)
        unTOLs.sort()
        tol = unTOLs[np.argmax(np.bincount(np.digitize(TOLs, unTOLs)))-1]
    else:
        tol = kwargs.pop("tol")
    fnNorm = kwargs.pop('fnNorm')
    from scipy.stats import norm
    x_data = np.array([itr.calcEg() for _, itr in enum_iter(runs, filteritr) if
                       itr.TOL == tol])
    x = x_data - np.mean(x_data, axis=0)
    try:
        x = x.astype(np.float)
        if len(x.shape) != 1:
            raise Exception("Not correct size")
    except:
        __plot_failed(ax)
        import warnings
        warnings.warn("QQ plots require the object to implement __float__")
        return

    x /= np.std(x)
    ec = ECDF(x)
    ax.set_xlabel(r'Empirical CDF')
    ax.set_ylabel("Normal CDF")

    plotObj = []
    ax.set_xlim([0, 1.])
    ax.set_ylim([0, 1.])
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    plotObj.append(ax.scatter(norm.cdf(x), ec(x), *args, **kwargs))
    if Ref_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(rate=1, **Ref_kwargs)))
    return plotObj[0].get_offsets(), plotObj


def __add_legend(ax, handles=None, labels=None, alpha=0.5,
                 outside=None, loc='best', *args, **kwargs):
    if not handles:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
    if outside is not None and len(handles) >= outside:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(handles, labels, loc='center left', fancybox=False,
                  frameon=False, shadow=False,
                  bbox_to_anchor=(1, 0.5)).draggable(True)
    else:
        ax.legend(handles, labels, loc=loc, fancybox=True,
                  shadow=True).draggable(True)


def __formatMIMCRate(rate, log_rate, lbl_base=r"\textrm{TOL}", lbl_log_base=None):
    txt_rate = '{:.2g}'.format(rate)
    txt_log_rate = '{:.2g}'.format(log_rate)
    lbl_log_base = lbl_log_base or "{}^{{-1}}".format(lbl_base)
    label = lbl_base
    if txt_rate != "1":
        label += r'^{{ {} }}'.format(txt_rate)
    if txt_log_rate != "0":
        label += r'\log\left({}\right)^{{ {} }}'.format(lbl_log_base,
                                                        txt_log_rate)
    return (lambda x, r=rate, lr=log_rate: x**r * np.abs(np.log(x))**lr), \
        "${}$".format(label)


def __plot_failed(ax):
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax.text(0.5*(left+right), 0.5*(bottom+top), 'PLOTTING\nFAILED',
            horizontalalignment='center', verticalalignment='center',
            rotation=45, fontsize=60, color='red', alpha=0.5,
            transform=ax.transAxes)

def __plot_except(ax):
    __plot_failed(ax)

    import traceback
    print('-----------------------------------------------------')
    traceback.print_exc(limit=None)
    print('-----------------------------------------------------')

    raise

@public
def genPDFBooklet(runs, fileName=None, exact=None, **kwargs):
    import matplotlib.pyplot as plt


    filteritr = kwargs.pop("filteritr", filteritr_convergent)

    TOLs_count = len(np.unique([itr.TOL for _, itr in enum_iter(runs, filteritr)]))
    convergent_count = len(np.unique([itr.TOL for _, itr in enum_iter(runs, filteritr_convergent)]))
    iters_count = np.sum([len(r.iters) for r in runs])
    verbose = kwargs.pop('verbose', False)
    if verbose:
        def print_msg(*args):
            print(*args)
    else:
        def print_msg(*args):
            return

    if "params" in kwargs:
        params = kwargs.pop("params")
        fn = kwargs.pop("fn")
    else:
        maxTOL = np.max([r.db_data.finalTOL for r in runs])
        params = next(r.params for r in runs if r.db_data.finalTOL == maxTOL)
        fn = next(r.fn for r in runs if r.db_data.finalTOL == maxTOL)

    max_dim = np.max([np.max(r.last_itr.lvls_max_dim()) for r in runs])
    any_bayesian = np.any([r.params.bayesian for r in runs])

    legend_outside = kwargs.pop("legend_outside", 5)

    has_gamma_rate = hasattr(params, 'gamma')
    has_w_rate = hasattr(params, 'w')
    has_s_rate = hasattr(params, 's')
    has_beta = hasattr(params, 'beta')

    if exact is None:
        exact = estimate_exact(runs)
        print("Estimated exact value is {:.14f}".format(exact))

    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rc('font', **{'family': 'normal', 'weight': 'demibold',
                      'size': 15})

    figures = []
    def add_fig():
        figures.append(plt.figure())
        return figures[-1].gca()

    if (TOLs_count > 1):
        print_msg("plotErrorsVsTOL")
        ax = add_fig()
        try:
            plotErrorsVsTOL(ax, runs, exact=exact, filteritr=filteritr,
                            relative=True, fnNorm=fn.Norm,
                            ErrEst_kwargs={'label': 'Error Estimate'},
                            Ref_kwargs={'ls': '--', 'c':'k', 'label': 'TOL'},
                            num_kwargs={'color': 'r'})
        except:
            __plot_except(ax)

    print_msg("plotWorkVsMaxError")
    ax = add_fig()
    try:
        plotWorkVsMaxError(ax, runs, exact=exact, filteritr=filteritr,
                           relative=True, fnNorm=fn.Norm, fmt='-*',
                           ErrEst_kwargs={'fmt': '--*', 'label': 'Error Estimate'},
                           Ref_kwargs={'ls': '--', 'c':'k', 'label': '{rate:.2g}'})
    except:
        __plot_except(ax)

    print_msg("plotWorkVsLvlStats")
    ax = add_fig()
    try:
        plotWorkVsLvlStats(ax, runs, filteritr=filteritr,
                           fmt='-ob', label='Max dim.',
                           active_kwargs={'fmt': '-*g', 'label': 'Max active dim.'},
                           maxlvl_kwargs={'fmt': '-sr', 'label': 'Max level'})
    except:
        __plot_except(ax)

    if (convergent_count > 10):   # Need at least 10 plots for this plot to be significant
        print_msg("plotErrorsQQ")
        ax = add_fig()
        try:
            # This plot only makes sense for convergent plots
            plotErrorsQQ(ax, runs, filteritr=filteritr_convergent,
                         fnNorm=fn.Norm,
                         Ref_kwargs={'ls': '--', 'c': 'k'})
        except:
            __plot_except(ax)


    if (TOLs_count > 1):
        print_msg("plotTimeVsTOL")
        ax_time = add_fig()
        try:
            data_time, _ = plotTimeVsTOL(ax_time, runs, label="MIMC",
                                         filteritr=filteritr,
                                         MC_kwargs=None if max_dim > 1
                                         else {"label": "MC Estimate", "fmt": "--r"})
        except:
            __plot_except(ax_time)

        print_msg("plotTimeVsTOL")
        ax_est = add_fig()
        try:
            data_est, _ = plotTimeVsTOL(ax_est, runs, label="MIMC",
                                        work_estimate=True,
                                        MC_kwargs= None if max_dim > 1 else {"label": "MC Estimate", "fmt":
                                                                             "--r"})
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
                                                label=label))
                ax_est.add_line(FunctionLine2D(fn=func,
                                               data=data_est,
                                               linestyle='--', c='k',
                                               label=label))
        except:
            __plot_except(ax_est)

    def formatPower(rate):
        rate = "{:.2g}".format(rate)
        if rate == "-1":
            return "-"
        elif rate == "1":
            return ""
        return rate

    def getLevelRate(rate):
        if has_beta:
            func = lambda x, r=rate, b=params.beta[0]: b ** (r*x)
            label = r'${:.2g}^{{ {}\ell }}$'.format(params.beta[0],
                                                    formatPower(rate))
        else:
            func = lambda x, r=rate: np.exp(-r*x)
            label = r'$\exp({}\ell)$'.format(formatPower(rate))
        return func, label

    lvl_funcs = [[0, False, False, plotTimeVsLvls, np.array(params.gamma)
                  if has_gamma_rate else None],
                 [1, True, False, plotExpectVsLvls, -np.array(params.w)
                  if has_w_rate else None],
                 [2, True, any_bayesian, plotVarVsLvls, -np.array(params.s)
                  if has_s_rate else None],
                 [3, False, False, plotSkewnessVsLvls, None],
                 [4, False, False, plotKurtosisVsLvls, None]]
    directions = np.eye(np.minimum(5, max_dim), dtype=np.int).tolist()
    cur = np.array(directions[0])
    for i in range(1, len(directions)):
        cur += np.array(directions[i])
        directions.append(cur.tolist())

    max_moment = runs[0].last_itr.psums_delta.shape[1]
    for min_moment, plotFine, plotEstimate, plotFunc, rate in lvl_funcs:
        if min_moment > max_moment:
            continue
        try:
            print_msg(plotFunc.__name__)
            ax = add_fig()
            add_rates = dict()
            from itertools import cycle
            markers = cycle(['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd'])
            linestyles = cycle(['--', '-.', '-', ':', '-'])
            cycler = ax._get_lines.prop_cycler
            for j, direction in enumerate(directions):
                mrk = next(markers)
                prop = next(cycler)
                labal = '$\Delta$' if len(directions)==1 \
                        else "$[{}]$".format(",".join(["0" if d==0
                                                       else ("" if d==1 else str(d))
                                                       +"\ell" for d in direction]))
                cur_kwargs = {'ax' : ax, 'runs': runs,
                              'linestyle' : '-',
                              'marker' : mrk,
                              'label': labal,
                              'direction' : direction}
                cur_kwargs.update(prop)
                if plotFine:
                    cur_kwargs['fine_kwargs'] = {'linestyle': '--',
                                                 'marker' : mrk}
                    cur_kwargs['fine_kwargs'].update(prop)

                if max_dim == 1 and plotEstimate:
                    cur_kwargs['estimate_kwargs'] = {'linestyle': ':',
                                                     'marker' : mrk,
                                                     'label' : 'Corrected estimate'}

                line_data, _ = plotFunc(fnNorm=fn.Norm, **cur_kwargs)
                if rate is None:
                    continue
                ind = np.nonzero(np.array(direction) != 0)[0]
                if np.all(ind < len(rate)):
                    add_rates[np.sum(rate[ind])] = line_data

            for j, r in enumerate(sorted(add_rates.keys(), key=lambda x:
                                         np.abs(x))):
                func, label = getLevelRate(r)
                ax.add_line(FunctionLine2D(fn=func, data=add_rates[r][1:, :],
                                           linestyle=next(linestyles),
                                           c='k', label=label))
        except:
            __plot_except(ax)

    # print_msg("plotTotalWorkVsLvls")
    # ax = add_fig()
    # try:
    #     plotTotalWorkVsLvls(ax, runs,
    #                         fmt='-o',  filteritr=filteritr,
    #                         label_fmt="${:.2g}$",
    #                         max_TOLs=5, max_dim=max_dim)
    # except:
    #     __plot_except(ax)

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
                label = r'${}\log\left(\textrm{{TOL}}^{{-1}}\right)$'.format(formatPower(rate))
                ax.add_line(FunctionLine2D(fn=lambda x, r=rate: -rate*np.log(x),
                                           data=line_data,
                                           log_data=False,
                                           linestyle='--', c='k',
                                           label=label))
        except:
            __plot_except(ax)

        print_msg("plotThetaVsTOL")
        ax = add_fig()
        try:
            plotThetaVsTOL(ax, runs)
            if max_dim == 1 and has_s_rate and has_w_rate and has_gamma_rate:
                chi = params.s[0]/params.gamma[0]
                eta = params.w[0]/params.gamma[0]
                plotThetaRefVsTOL(ax, runs, filteritr=filteritr, eta=eta, chi=chi, fmt='--k')
        except:
            __plot_except(ax)

    if fileName is not None:
        if verbose:
            print("Saving file")
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(fileName) as pdf:
            for fig in figures:
                for ax in fig.axes:
                    __add_legend(ax, outside=legend_outside)
                pdf.savefig(fig)
    return figures

def run_program():
    from . import db as mimcdb
    from . import plot as miplot
    from . import test

    import argparse
    import warnings
    import os
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
                            help="Database Tag")
        parser.add_argument("-qoi_exact", type=float, action="store",
                            help="Exact value")
        parser.add_argument("-only_final", type='bool', action="store",
                            default=True, help="Plot only final iterations")
        parser.add_argument("-o", type=str,
                            action="store", help="Output file")
        parser.add_argument("-cmd", type=str, action="store",
                            help="Command to execute after plotting")
        parser.add_argument("-verbose", type='bool', action="store",
                            default=False)
        parser.add_argument("-all_itr", type='bool', action="store",
                            default=False)
        parser.add_argument("-done_flag", type=int, nargs='+',
                            action="store", default=None)

    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    args = test.parse_known_args(parser)
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
        args.o = args.db_tag + ".pdf"
    db = mimcdb.MIMCDatabase(**db_args)
    if args.db_tag is None:
        warnings.warn("You did not select a database tag!!")
    if args.verbose:
        print("Reading data")

    run_data = db.readRuns(tag=args.db_tag, done_flag=args.done_flag)
    if len(run_data) == 0:
        raise Exception("No runs!!!")
    if args.verbose:
        print("Plotting data")

    miplot.genPDFBooklet(run_data,
                         fileName=args.o,
                         exact=args.qoi_exact, verbose=args.verbose,
                         filteritr = filteritr_all if args.all_itr
                         else filteritr_convergent)

    if args.cmd is not None:
        os.system(args.cmd.format(args.o))
