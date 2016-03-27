import numpy as np
import matplotlib.pylab as plt
from . import mimc
from matplotlib.ticker import MaxNLocator

__all__ = []


def public(sym):
    __all__.append(sym.__name__)
    return sym


@public
class FunctionLine2D(plt.Line2D):
    def __init__(self, fn, data=None, **kwargs):
        self.flip = kwargs.pop('flip', False)
        self.fn = fn
        if data is not None:
            x = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            if len(x) > 0 and len(y) > 0:
                const = [np.mean(y/fn(x)), 0]
                # const = np.polyfit(fn(x), y, 1)
                # print(const, np.mean(y/fn(x)))
                self.fn = lambda x, cc=const, ff=fn: cc[0] * ff(x) + cc[1]

        super(FunctionLine2D, self).__init__([], [], **kwargs)

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
    def ExpLine(rate, const=1, data=None, **kwargs):
        return FunctionLine2D(lambda x, r=rate: const*np.array(x)**r,
                              data=data, **kwargs)


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
        y.append([np.min(all_y), np.mean(all_y), np.max(all_y)])
        x.append(k)
    return np.array(x), np.array(y)


@public
def plotErrorsVsTOL(ax, runs_data, *args, **kwargs):
    """Plots Errors vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes

runs_data is a list
run_data[i] is another class
run_data[i].TOL
run_data[i].finalTOL
run_data[i].creation_date
run_data[i].iteration_index
run_data[i].total_iterations
run_data[i].totalTime
run_data[i].run is an instance of mimc.MIMCRun
run_data[i].run.data is an instance of mimc.MIMCData
"""

    exact = kwargs.pop('exact', None)
    if exact is None:
        # Calculate mean based on data
        minTOL = np.min([r.finalTOL for r in runs_data])
        exact = np.mean([r.run.data.calcEg() for r in runs_data if
                         r.finalTOL == minTOL])

    xy = np.array([[r.finalTOL, np.abs(exact - r.run.data.calcEg()),
                    r.run.totalErrorEst()]
                   for r in runs_data])
    relative_error = kwargs.pop('relative', True)
    if relative_error:
        xy[:, 1:] = xy[:, 1:]/exact
    TOLs, error_est = __get_stats(xy, staton=2)

    plotObj = []

    ax.set_xlabel('TOL')
    ax.set_ylabel('Errors')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ErrEst_kwargs = kwargs.pop('ErrEst_kwargs')
    TOLRef_kwargs = kwargs.pop('TOLRef_kwargs')
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
    if TOLRef_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(1, const=1./exact
                                                          if relative_error
                                                          else 1.,
                                                          **TOLRef_kwargs)))
    return xy[sel, :2], plotObj


def __calc_moments(runs_data, seed=None, direction=None):
    dim = runs_data[0].run.data.dim
    seed = np.array(seed) if seed is not None else np.zeros(dim, dtype=np.uint32)
    direction = np.array(direction) if direction is not None else np.ones(dim, dtype=np.uint32)

    psums = np.zeros((0, 2))
    Tl = np.zeros(0)
    M = np.zeros(0)
    for curRun in runs_data:
        cur = seed
        inds = []
        while True:
            ii = next((i for i, l in enumerate(curRun.run.data.lvls)
                       if np.all(l == cur)), None)
            if ii is None:
                break
            inds.append(ii)
            cur = cur + direction
        L = len(inds)
        if L > len(M):
            psums.resize((L, 2), refcheck=False)
            M.resize(L, refcheck=False)
            Tl.resize(L, refcheck=False)
        psums[:L, :] += curRun.run.data.psums[inds, :2]
        M[:L] += curRun.run.data.M[inds]
        Tl[:L] += curRun.run.data.t[inds]

    El = psums[:, 0]/M
    Vl = psums[:, 1]/M - El**2
    Tl /= M
    return El, Vl, Tl, M

def __normalize_fmt(args, kwargs):
    if "fmt" in kwargs:        # Normalize behavior of errorbar() and plot()
        args = (kwargs.pop('fmt'), ) + args
    return args, kwargs

@public
def plotExpectVsLvls(ax, runs_data, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$E_\ell$')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    El, Vl, _, M = __calc_moments(runs_data,
                                  seed=kwargs.pop('seed', None),
                                  direction=kwargs.pop('direction', None))
    errorBar = ax.errorbar(np.arange(0, len(El)), np.abs(El), *args,
                           yerr=3*np.sqrt(np.abs(Vl/M)), **kwargs)
    return errorBar[0].get_xydata(), [errorBar]


@public
def plotVarVsLvls(ax, runs_data, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    args, kwargs = __normalize_fmt(args, kwargs)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$V_\ell$')
    ax.set_yscale('log')
    _, Vl, _, _ = __calc_moments(runs_data,
                                 seed=kwargs.pop('seed', None),
                                 direction=kwargs.pop('direction', None))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    line = ax.plot(np.arange(0, len(Vl)), Vl, *args, **kwargs)
    return line[0].get_xydata(), [line]


@public
def plotTimeVsLvls(ax, runs_data, *args, **kwargs):
    """Plots Time vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    args, kwargs = __normalize_fmt(args, kwargs)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')
    _, _, Tl, _ = __calc_moments(runs_data,
                                 seed=kwargs.pop('seed', None),
                                 direction=kwargs.pop('direction', None))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    line = ax.plot(np.arange(0, len(Tl)), Tl, *args, **kwargs)
    return line[0].get_xydata(), [line]


@public
def plotTimeVsTOL(ax, runs_data, *args, **kwargs):
    """Plots Tl vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    work_estimate = kwargs.pop("work_estimate", False)
    if kwargs.pop("real_time", False):
        if work_estimate:
            raise ValueError("real_time or work_estimate cannot be both True")
        xy = [[r.TOL, r.totalTime] for r in runs_data]
    elif work_estimate:
        xy = [[r.TOL, np.sum(r.run.data.M*r.run.Wl_estimate),
               r.run.Wl_estimate[-1] *
               np.maximum(r.run.params.M0, r.run.params.Ca *
                          r.run.all_data.calcVl()[0] * r.TOL**-2.)]
              for r in runs_data]
    else:
        xy = [[r.TOL, np.sum(r.run.data.t),
               r.run.all_data.calcTl()[-1] *
               np.maximum(r.run.params.M0, r.run.params.Ca *
                          r.run.all_data.calcVl()[0] * r.TOL**-2.)]
              for r in runs_data]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('TOL')
    if work_estimate:
        ax.set_ylabel('Work estimate')
    else:
        ax.set_ylabel('Time (s)')

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
def plotLvlsNumVsTOL(ax, runs_data, *args, **kwargs):
    """Plots L vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    summary = np.array([[r.TOL,
                         np.max([np.sum(l) for l in r.run.data.lvls])]
                        for r in runs_data])

    ax.set_xscale('log')
    ax.set_xlabel('TOL')
    ax.set_ylabel(r'$L$')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    scatter = ax.scatter(summary[:, 0], summary[:, 1], *args, **kwargs)
    return summary, [scatter]


@public
def plotThetaVsTOL(ax, runs_data, *args, **kwargs):
    """Plots theta vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    summary = np.array([[r.TOL, r.run.Q.theta] for r in runs_data])

    ax.set_xscale('log')
    ax.set_xlabel('TOL')
    ax.set_ylabel(r'$\theta$')
    ax.set_ylim([0, 1.])
    scatter = ax.scatter(summary[:, 0], summary[:, 1], *args, **kwargs)
    return summary, [scatter]


@public
def plotThetaRefVsTOL(ax, runs_data, eta, chi, *args, **kwargs):
    """Plots theta vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    El = np.abs(__calc_moments(runs_data)[0])
    L = lambda r: np.max([np.sum(l) for l in r.run.data.lvls])
    if chi == 1:
        summary = np.array([[r.TOL,
                             (1. + (1./(2.*eta))*1./(L(r)+1.))**-1,
                             1-El[L(r)]/r.TOL]
                            for r in runs_data])
    else:
        summary = np.array([[r.TOL,
                             (1. + (1./(2.*eta))*(1.-chi)/(1.-chi**(L(r)+1.)))**-1,
                             1-El[L(r)]/r.TOL]
                            for r in runs_data])
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
def plotErrorsQQ(ax, runs_data, *args, **kwargs): #(runs, tol, marker='o', color="b", fig=None, label=None):
    """Plots Normal vs Empirical CDF of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    if "tol" not in kwargs:
        TOLs = [r.TOL for r in runs_data]
        unTOLs = np.unique([r.TOL for r in runs_data])
        unTOLs.sort()
        tol = unTOLs[np.argmax(np.bincount(np.digitize(TOLs, unTOLs)))-1]
    else:
        tol = kwargs.pop("tol")
    from scipy.stats import norm
    x = [r.run.data.calcEg() for r in runs_data if r.TOL == tol]
    x = np.array(x)
    x = (x - np.mean(x)) / np.std(x)
    ec = ECDF(x)
    ax.set_xlabel(r'Empirical CDF')
    ax.set_ylabel("Normal CDF")

    plotObj = []
    ax.set_xlim([0, 1.])
    ax.set_ylim([0, 1.])
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    plotObj.append(ax.scatter(norm.cdf(x), ec(x), *args, **kwargs))
    if Ref_kwargs is not None:
        plotObj.append(ax.add_line(FunctionLine2D.ExpLine(1, **Ref_kwargs)))
    return plotObj[0].get_offsets(), plotObj


def __add_legend(ax, handles=None, labels=None, alpha=0.5,
                 outside=False, loc='best', *args, **kwargs):
    if not handles:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
    if outside:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(handles, labels, loc='center left', fancybox=True,
                  shadow=True, bbox_to_anchor=(1, 0.5)).draggable(True)
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


@public
def genPDFBooklet(runs_data, fileName=None, exact=None, **kwargs):
    import matplotlib.pyplot as plt

    if "params" in kwargs:
        params = kwargs.pop("params")
    else:
        maxTOL = np.max([r.TOL for r in runs_data])
        params = next(r.run.params for r in runs_data if r.TOL == maxTOL)

    dim = params.dim
    has_gamma_rate = hasattr(params, 'gamma')
    has_w_rate = hasattr(params, 'w')
    has_s_rate = hasattr(params, 's')
    has_beta = hasattr(params, 'beta')

    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rc('font', **{'family': 'normal', 'weight': 'demibold',
                      'size': 15})

    figures = []
    def add_fig():
        figures.append(plt.figure())
        return figures[-1].gca()

    ax = add_fig()
    plotErrorsVsTOL(ax, runs_data, exact=exact,
                    relative=True,
                    ErrEst_kwargs={'label': 'Error Estimate'},
                    TOLRef_kwargs={'linestyle': '--', 'c': 'k', 'label': 'TOL'})

    ax = add_fig()
    plotErrorsQQ(ax, runs_data, Ref_kwargs={'linestyle': '--', 'c': 'k'})

    ax = add_fig()
    data_mimc, _ = plotTimeVsTOL(ax, runs_data, label="MIMC",
                                 MC_kwargs={"label": "MC Estimate", "fmt": "--r"})
    ax_est = add_fig()
    data_mc, _ = plotTimeVsTOL(ax_est, runs_data, label="MIMC",
                               work_estimate=True,
                               MC_kwargs={"label": "MC Estimate", "fmt":
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
        ax.add_line(FunctionLine2D(func, data=data_mimc,
                                   linestyle='--', c='k',
                                   label=label))
        ax_est.add_line(FunctionLine2D(func,
                                       data=data_mc,
                                       linestyle='--', c='k',
                                       label=label))

    def formatPower(rate):
        rate = "{:.2g}".format(rate)
        if rate == "-1":
            return "-"
        elif rate == "1":
            return ""
        return rate

    def getLevelRate(rates):
        assert(len(rates) == 1)
        rate = np.sum(rates)
        if has_beta:
            func = lambda x, r=rate, b=params.beta[0]: b ** (r*x)
            label = r'${:.2g}^{{ {}\ell }}$'.format(params.beta[0],
                                                    formatPower(rate))
        else:
            func = lambda x, r=rate: np.exp(-r*x)
            label = r'$\exp({}\ell)$'.format(formatPower(rate))
        return func, label

    lvl_funcs = [[plotExpectVsLvls, -np.array(params.w) if has_w_rate else None],
                 [plotVarVsLvls, -np.array(params.s) if has_s_rate else None],
                 [plotTimeVsLvls, np.array(params.gamma) if has_gamma_rate else None]]

    for plotFunc, rate in lvl_funcs:
        if dim != 1:
            continue
        ax = add_fig()
        line_data, _ = plotFunc(ax, runs_data, fmt='-o')
        if rate is None:
            continue
        assert(dim == 1)
        func, label = getLevelRate(np.array(np.array(rate)))
        ax.add_line(FunctionLine2D(func,
                                   data=line_data,
                                   linestyle='--', c='k',
                                   label=label))

    ax = add_fig()
    line_data, _ = plotLvlsNumVsTOL(ax, runs_data)
    if has_beta and has_w_rate and has_gamma_rate:
        if has_beta:
            rate = 1./np.min(np.array(params.w) * np.log(params.beta))
        else:
            rate = 1./np.min(np.array(params.w))
        label = r'${}\log\left(\textrm{{TOL}}^{{-1}}\right)$'.format(formatPower(rate))
        ax.add_line(FunctionLine2D(lambda x, r=rate: -rate*np.log(x),
                                   data=line_data,
                                   linestyle='--', c='k',
                                   label=label))

    ax = add_fig()
    plotThetaVsTOL(ax, runs_data)
    if dim == 1 and has_s_rate and has_w_rate and has_gamma_rate:
        chi = params.s[0]/params.gamma[0]
        eta = params.w[0]/params.gamma[0]
        plotThetaRefVsTOL(ax, runs_data, eta=eta, chi=chi, fmt='--k')

    if fileName is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(fileName) as pdf:
            for fig in figures:
                __add_legend(fig.gca())
                pdf.savefig(fig)

    return figures
