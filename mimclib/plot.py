import numpy as np

def plotTOLvsErrors(ax, runs_data, exact=None, **kwargs):
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
    return ax.scatter(xy[sel, 0], xy[sel, 1], **kwargs)
