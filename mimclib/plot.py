import numpy as np

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

def plotTOLvsErrors(ax, runs_data, exact, *args, **kwargs):
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
    if exact is None:
        # Calculate mean based on data
        minTOL = np.min([r.TOL for r in runs_data])
        exact = np.mean([r.run.data.calcEg() for r in runs_data if
                         r.TOL == minTOL])

    xy = np.array([[r.TOL, np.abs(exact - r.run.data.calcEg())] for r
                   in runs_data])

    ax.set_yscale('log')
    ax.set_xscale('log')
    sel = np.logical_and(np.isfinite(xy[:, 1]), xy[:, 1] >=
                         np.finfo(float).eps)
    return ax.scatter(xy[sel, 0], xy[sel, 1], *args, **kwargs)
  
def plotExpectVsLvls(ax, runs_data, *args, **kwargs):
    """Plots El, Vl vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    maxL = np.max([len(r.run.data.lvls) for r in runs_data])
    max_dim = np.max([r.run.data.dim for r in runs_data])
    if max_dim > 1:
        raise Exception("This function is only for 1D MIMC")
    psums = np.zeros((maxL, 2))
    M = np.zeros(maxL)
    for r in runs_data:
        L = len(r.run.data.lvls)
        psums[:L, :] += r.run.data.psums[:, :1]
        M[:L] += r.run.data.M

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$E_\ell$')
    ax.set_yscale('log')
    El = psums[:, 0]/M
    Vl = psums[:, 1]/M - El**2
    return ax.errorbar(np.arange(0, maxL), El,
                       yerr=3*np.sqrt(Vl/M), *args, **kwargs)

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
        
    ax.set_yscale('log')
    return ax.plot(np.arange(0, maxL), Tl/M,
                   *args, **kwargs)

def plotTimeVsTOL(ax, runs_data, *args, **kwargs):
    """Plots Tl vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    real_time = False
    if "real_time" in kwargs:
        real_time = kwargs["real_time"]
    if real_time:
        TotalTime = [np.sum(r.totalTime) for r in runs_data]
        minTime = np.min([r.totalTime for r in runs_data])
        maxTime = np.max([r.totalTime for r in runs_data])
    else:
        TotalTime = [np.sum(r.run.data.t) for r in runs_data]
        minTime = np.min([r.run.data.t for r in runs_data])
        maxTime = np.max([r.run.data.t for r in runs_data])
    TOL = [r.TOL for r in runs_data]
    N = [len(r.TOL) for r in runs_data]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('TOL')
    ax.set_ylabel('Time (s)')
    return ax.plot(TOL, TotalTime/N,
                   yerr=(maxTime-minTime), *args, **kwargs)

def plotLvlsVsTOL(ax, runs_data, *args, **kwargs):
    """Plots L vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    L = [len(r.run.data.lvls) for r in runs_data]
    TOL = [r.TOL for r in runs_data]
    max_dim = np.max([r.run.data.dim for r in runs_data])
    if max_dim > 1:
        raise Exception("This function is only for 1D MIMC")
        
    ax.set_xscale('log')
    return ax.scatter(TOL, L, *args, **kwargs)
  
def plotErrorsQQ(ax, runs_data, *args, **kwargs) #(runs, tol, marker='o', color="b", fig=None, label=None):
    """Plots Normal vs Empirical CDF of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    from scipy.stats import norm
    x = [r.run.data.calcEg() for r in runs_data if r.TOL == kwargs["tol"]]
    x = np.array(x)
    x = (x - np.mean(x)) / np.std(x)
    ec = ECDF(x)
    ax.set_xlabel(r'Empirical CDF')
    ax.set_ylabel("Normal CDF")
    return ax.scatter(norm.cdf(x), ec(x), *args, **kwargs)
