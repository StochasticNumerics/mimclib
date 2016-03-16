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
    TOLs, error_est = __get_stats(xy, staton=2)

    relative_error = kwargs.pop('relative', True)
    if not kwargs.pop('no_ref', False):
        ax.add_line(FunctionLine2D.ExpLine(1, const=1./exact if relative_error else 1.,
                                           linestyle='--', c='k',
                                           label='TOL'))

    ax.set_xlabel('TOL')
    ax.set_ylabel('Errors')
    ax.set_yscale('log')
    ax.set_xscale('log')
    sel = np.logical_and(np.isfinite(xy[:, 1]), xy[:, 1] >=
                         np.finfo(float).eps)

    if relative_error:
        error_est = error_est/exact
        xy[sel, 1] = xy[sel, 1]/exact
    ax.errorbar(TOLs, error_est[:, 1], yerr=[error_est[:, 1]-error_est[:, 0],
                                             error_est[:, 2]-error_est[:, 1]],
                label="Error Estimate")
    return ax.scatter(xy[sel, 0], xy[sel, 1], *args, **kwargs)


def __calc_moments(runs_data):
    maxL = np.max([len(r.run.data.lvls) for r in runs_data])
    max_dim = np.max([r.run.data.dim for r in runs_data])
    if max_dim > 1:
        raise Exception("This function is only for 1D MIMC")
    psums = np.zeros((maxL, 2))
    M = np.zeros(maxL)
    for r in runs_data:
        L = len(r.run.data.lvls)
        psums[:L, :] += r.run.data.psums[:, :2]
        M[:L] += r.run.data.M

    El = psums[:, 0]/M
    Vl = psums[:, 1]/M - El**2
    return El, Vl, M


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
    El, Vl, M = __calc_moments(runs_data)
    return ax.errorbar(np.arange(0, len(El)), El, *args,
                       yerr=3*np.sqrt(np.abs(Vl/M)), **kwargs)


@public
def plotVarVsLvls(ax, runs_data, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$V_\ell$')
    ax.set_yscale('log')
    _, Vl, _ = __calc_moments(runs_data)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax.plot(np.arange(0, len(Vl)), Vl, *args, **kwargs)


@public
def plotTimeVsLvls(ax, runs_data, *args, **kwargs):
    """Plots Time vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    maxL = np.max([len(r.run.data.lvls) for r in runs_data])
    max_dim = np.max([r.run.data.dim for r in runs_data])
    if max_dim > 1:
        raise Exception("This function is only for 1D MIMC")
    Tl = np.zeros(maxL)
    M = np.zeros(maxL)
    for r in runs_data:
        L = len(r.run.data.lvls)
        Tl[:L] += r.run.data.t
        M[:L] += r.run.data.M

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax.plot(np.arange(0, maxL), Tl/M, *args, **kwargs)


@public
def plotTimeVsTOL(ax, runs_data, *args, **kwargs):
    """Plots Tl vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    real_time = kwargs.pop("real_time", False)
    work_estimate = kwargs.pop("work_estimate", False)
    if kwargs.pop("real_time", False):
        if work_estimate:
            raise ValueError("real_time or work_estimate cannot be both True")
        xy = [[r.TOL, r.totalTime] for r in runs_data]
    elif work_estimate:
        xy = [[r.TOL, np.sum(r.run.data.M*r.run.Wl_estimate),
               r.run.Wl_estimate[-1] * r.run.params.Ca *
               r.run.data.calcVl()[0] * r.TOL**-2.]
              for r in runs_data]
    else:
        xy = [[r.TOL, np.sum(r.run.data.t), r.run.data.t[-1] *
               r.run.params.Ca * r.run.data.calcVl()[0] * r.TOL**-2.]
              for r in runs_data]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('TOL')
    if work_estimate:
        ax.set_ylabel('Work estimate')
    else:
        ax.set_ylabel('Time (s)')

    if "MC_kwargs" in kwargs:
        TOLs, times = __get_stats(xy, staton=2)
        ax.errorbar(TOLs, times[:, 1], *args,
                    yerr=[times[:, 1]-times[:, 0],
                          times[:, 2]-times[:, 1]],
                    **kwargs.pop("MC_kwargs"))

    TOLs, times = __get_stats(xy)
    return ax.errorbar(TOLs, times[:, 1], *args,
                       yerr=[times[:, 1]-times[:, 0],
                             times[:, 2]-times[:, 1]],
                       **kwargs)


@public
def plotLvlsNumVsTOL(ax, runs_data, *args, **kwargs):
    """Plots L vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    summary = np.array([[r.TOL, len(r.run.data.lvls), r.run.data.dim]
                        for r in runs_data])
    if np.max(summary[:, 2]) > 1:
        raise Exception("This function is only for 1D MIMC")

    ax.set_xscale('log')
    ax.set_xlabel('TOL')
    ax.set_ylabel(r'$L$')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax.scatter(summary[:, 0], summary[:, 1], *args, **kwargs)


@public
def plotThetaVsTOL(ax, runs_data, *args, **kwargs):
    """Plots theta vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    summary = np.array([[r.TOL,
                         r.run.Q.theta if hasattr(r.run.Q, 'theta') else
                         r.run._calcTheta(r.TOL, r.run.bias)]
                        for r in runs_data])

    ax.set_xscale('log')
    ax.set_xlabel('TOL')
    ax.set_ylabel(r'$\theta$')
    ax.set_ylim([0, 1.])
    return ax.scatter(summary[:, 0], summary[:, 1], *args, **kwargs)


@public
def plotErrorsQQ(ax, runs_data, *args, **kwargs): #(runs, tol, marker='o', color="b", fig=None, label=None):
    """Plots Normal vs Empirical CDF of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    tol = kwargs.pop("tol", np.min([r.TOL for r in runs_data]))
    from scipy.stats import norm
    x = [r.run.data.calcEg() for r in runs_data if r.TOL == tol]
    x = np.array(x)
    x = (x - np.mean(x)) / np.std(x)
    ec = ECDF(x)
    ax.set_xlabel(r'Empirical CDF')
    ax.set_ylabel("Normal CDF")

    if not kwargs.pop('no_ref', False):
        ax.add_line(FunctionLine2D.ExpLine(1, linestyle='--', c='k'))
    ax.set_xlim([0, 1.])
    ax.set_ylim([0, 1.])
    return ax.scatter(norm.cdf(x), ec(x), *args, **kwargs)


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
    lbl_log_base = lbl_log_base or "{}^{{-1}}".format(lbl_base)
    label = lbl_base
    if rate != 1:
        label += r'^{{ {:.2g} }}'.format(rate)
    if log_rate != 1:
        label += r'\log\left({}\right)^{{ {:.2g} }}'.format(lbl_log_base, log_rate)
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
    plotErrorsVsTOL(ax, runs_data, exact=exact)

    ax = add_fig()
    plotErrorsQQ(ax, runs_data)

    ax = add_fig()
    line = plotTimeVsTOL(ax, runs_data, label="MIMC",
                         MC_kwargs={"label": "MC Estimate", "fmt": "--r"})[0]
    ax_est = add_fig()
    line_est = plotTimeVsTOL(ax_est, runs_data, label="MIMC",
                             work_estimate=True,
                             MC_kwargs={"label": "MC Estimate", "fmt":
                                        "--r"})[0]
    if has_s_rate and has_gamma_rate and has_w_rate:
        s = np.array(params.s)
        w = np.array(params.w)
        gamma = np.array(params.gamma)
        if has_beta:
            s = s * np.log(params.beta)
            w = w * np.log(params.beta)
            gamma = gamma * np.log(params.beta)
        func, label = __formatMIMCRate(*mimc.calcMIMCRate(w, s, gamma))
        ax.add_line(FunctionLine2D(func, data=line.get_xydata(),
                                   linestyle='--', c='k',
                                   label=label))
        ax_est.add_line(FunctionLine2D(func,
                                       data=line_est.get_xydata(),
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

    ax = add_fig()
    line = plotExpectVsLvls(ax, runs_data, fmt='-o')[0]
    if has_w_rate:
        assert(dim == 1)
        func, label = getLevelRate(-np.array(params.w))
        ax.add_line(FunctionLine2D(func,
                                   data=line.get_xydata(),
                                   linestyle='--', c='k',
                                   label=label))

    ax = add_fig()
    line, = plotVarVsLvls(ax, runs_data, '-o')
    if has_beta and has_s_rate:
        assert(dim == 1)
        func, label = getLevelRate(-np.array(params.s))
        ax.add_line(FunctionLine2D(func,
                                   data=line.get_xydata(),
                                   linestyle='--', c='k',
                                   label=label))

    ax = add_fig()
    line, = plotTimeVsLvls(ax, runs_data, '-o')
    if has_beta and has_gamma_rate:
        assert(dim == 1)
        # TODO: The following if statement is for compatibility with
        # old data and should be removed
        func, label = getLevelRate(np.array([params.gamma] if
                                            np.isscalar(params.gamma)
                                            else params.gamma))
        ax.add_line(FunctionLine2D(func,
                                   data=line.get_xydata(),
                                   linestyle='--', c='k',
                                   label=label))

    ax = add_fig()
    line = plotLvlsNumVsTOL(ax, runs_data)
    if has_beta and has_w_rate and has_gamma_rate:
        assert(dim == 1)
        eta = np.array(params.w)
        if has_beta:
            eta = eta * np.log(params.beta)
        rate = 1./eta[0]
        label = r'${}\log\left(\textrm{{TOL}}^{{-1}}\right)$'.format(formatPower(rate))
        ax.add_line(FunctionLine2D(lambda x, r=rate: -rate*np.log(x),
                                   data=line.get_offsets(),
                                   linestyle='--', c='k',
                                   label=label))

    ax = add_fig()
    line = plotThetaVsTOL(ax, runs_data)

    if fileName is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(fileName) as pdf:
            for fig in figures:
                __add_legend(fig.gca())
                pdf.savefig(fig)

    return figures
