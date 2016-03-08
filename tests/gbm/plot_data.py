#!/usr/bin/python
import mimclib.db as mimcdb
import mimclib.plot as miplot
import numpy as np

import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.simplefilter('ignore', MatplotlibDeprecationWarning)

def addExtraArguments(parser):
    parser.register('type', 'bool', lambda v: v.lower() in ("yes", "true", "t", "1"))
    parser.add_argument("-db_name", type=str, default='mimc',
                        action="store", help="Database Name")
    parser.add_argument("-db_user", type=str, default=None,
                        action="store", help="Database User")
    parser.add_argument("-db_host", type=str, default='localhost',
                        action="store", help="Database Host")
    parser.add_argument("-db_tag", type=str, default="NoTag",
                        action="store", help="Database Tag")
    parser.add_argument("-qoi_exact", type=float, default=np.exp(1.),
                        action="store", help="Exact value")

    parser.add_argument("-o", type=str, default="out.pdf",
                        action="store", help="Output file")


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    args = parser.parse_known_args()[0]
    db = mimcdb.MIMCDatabase(db=args.db_name,
                             user=args.db_user,
                             host=args.db_host)

    run_data = db.readRunData(db.getRunDataIDs(tag=args.db_tag, done_flag=[1]))
    run_data = [d for d in run_data if d.iteration_index+1 ==
                d.total_iterations]
    if len(run_data) == 0:
        raise Exception("No runs!!!")
    miplot.genPDFBooklet(args.o, run_data, exact=args.qoi_exact)

# import mimclib.db as mimcdb
# import mimclib.plot as miplot

# def refErr(d):
#     return np.abs(np.exp(1.)-d.run.data.calcEg())/d.finalTOL

# def get_runs(tag):
#     import mimclib.db as mimcdb
#     db = mimcdb.MIMCDatabase(user="abdo")
#     run_data = db.readRunData(db.getRunDataIDs(tag=tag, done_flag=[1]))
#     run_data = [d for d in run_data if d.iteration_index+1 ==
#                 d.total_iterations]
#     return run_data
#     #[[d.totalTime, d.data_id, d.finalTOL, refErr(d)] for d in run_data if refErr(d)>3]

# def plot_Vl():
#     fig = plt.figure()
#     miplot.plotVarVsLvls(fig.gca(), runs, '-ob', label="Real Variance")
#     Vl_estimate = wrong.run._estimateBayesianVl()
#     fig.gca().plot(np.arange(0, len(Vl_estimate)), Vl_estimate, '-sr',
#                    label="Bayesian")
#     fig.gca().plot(np.arange(0, len(wrong.run.data.lvls)),
#                    wrong.run.data.calcVl(), '-*g', label="Sample")
#     plt.legend()
#     fig.show()
#     return fig

# def plot_El():
#     fig = plt.figure()
#     miplot.plotExpectVsLvls(fig.gca(), runs, fmt='-ob', label="Real Expectation")
#     hl = wrong.run.fnHierarchy(wrong.run.data.lvls)
#     fig.gca().plot(np.arange(0, len(wrong.run.data.lvls)),
#                    np.abs(wrong.run.Q.W)*hl, '-*g',
#                    label="Bayesian")
#     plt.legend()
#     fig.show()
#     return fig

if __name__ == "__main__":
    main()
