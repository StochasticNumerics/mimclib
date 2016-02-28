import numpy as np

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
  
def plotElVsLvls(ax, runs_data, *args, **kwargs):
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

    ax.set_yscale('log')
    El = psums[:, 0]/M
    Vl = psums[:, 1]/M - El**2
    return ax.errorbar(np.arange(0, maxL), El,
                       yerr=3*np.sqrt(Vl/M), *args, **kwargs)
  
def plotTlVsLvls(ax, runs_data, *args, **kwargs):
    """Plots Tl vs TOL of @runs_data, as
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

def plotLVsTl(ax, runs_data, *args, **kwargs):
    """Plots L vs TOL of @runs_data, as
returned by MIMCDatabase.readRunData()
ax is in instance of matplotlib.axes
"""
    L = [len(r.run.data.lvls) for r in runs_data]
    TOL = [r.TOL for r in runs_data]
    max_dim = np.max([r.run.data.dim for r in runs_data])
    if max_dim > 1:
        raise Exception("This function is only for 1D MIMC")
        
    ax.set_yscale('log')
    return ax.plot(L, TOL,
                   *args, **kwargs)
  
