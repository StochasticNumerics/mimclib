from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#warnings.filterwarnings('error')
import numpy as np
import warnings

__all__ = []


def public(sym):
    __all__.append(sym.__name__)
    return sym

try:
    from mpi4py import MPI
except:
    pass

@public
def output_process():
    try:
        return MPI.COMM_WORLD.rank == 0
    except:
        return True

@public
class ArgumentWarning(Warning):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message


def parse_known_args(parser, return_unknown=False):
    knowns, unknowns = parser.parse_known_args()
    for a in unknowns:
        if a.startswith('-'):
            warnings.warn(ArgumentWarning("Argument {} was not used!".format(a)))

    if return_unknown:
        return knowns, unknowns
    return knowns


def RunStandardTest(fnSampleLvl=None,
                    fnSampleAll=None,
                    fnAddExtraArgs=None,
                    fnInit=None,
                    fnItrDone=None,
                    fnSeed=np.random.seed, profCalc=None):
    import warnings
    import os.path
    import mimclib.mimc as mimc
    import mimclib.db as mimcdb
    warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                             "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                       lineno, cat.__name__, msg)
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    parser.register('type', 'bool',
                    lambda v: v.lower() in ("yes", "true", "t", "1"))
    parser.add_argument("-db_user", type=str,
                        action="store", help="Database User")
    parser.add_argument("-db_password", type=str,
                        action="store", help="Database password")
    parser.add_argument("-db_host", type=str,
                        action="store", help="Database Host")
    parser.add_argument("-db_engine", type=str, default='mysql',
                        action="store", help="Database Host")
    parser.add_argument("-db_tag", type=str, default="NoTag",
                        action="store", help="Database Tag")
    parser.add_argument("-db", type='bool', default=False,
                        action="store", help="Save in Database")
    parser.add_argument("-qoi_seed", type=int,
                        action="store", help="Seed for random generator")
    parser.add_argument("-db_name", type=str, action="store", help="")

    if fnAddExtraArgs is not None:
        fnAddExtraArgs(parser)
    mimc.MIMCRun.addOptionsToParser(parser)
    mimcRun = mimc.MIMCRun(**vars(parse_known_args(parser)))

    if fnSampleLvl is not None:
        fnSampleLvl = lambda inds, M, fn=fnSampleLvl: fn(mimcRun, inds, M)
        mimcRun.setFunctions(fnSampleLvl=fnSampleLvl)
    else:
        fnSampleAll = lambda lvls, M, moments, fn=fnSampleAll: \
                      fn(mimcRun, lvls, M, moments)
        mimcRun.setFunctions(fnSampleAll=fnSampleAll)

    import time
    tStart = time.clock()
    if not hasattr(mimcRun.params, 'qoi_seed'):
        mimcRun.params.qoi_seed = np.random.randint(2**32-1)

    if fnInit is not None:
        res = fnInit(mimcRun)
        if res is not None and res < 0:
            return res

    if fnSeed is not None:
        fnSeed(mimcRun.params.qoi_seed)

    if mimcRun.params.db and output_process():
        db_args = {}
        if hasattr(mimcRun.params, "db_user"):
            db_args["user"] = mimcRun.params.db_user
        if hasattr(mimcRun.params, "db_password"):
            db_args["passwd"] = mimcRun.params.db_password
        if hasattr(mimcRun.params, "db_host"):
            db_args["host"] = mimcRun.params.db_host
        if hasattr(mimcRun.params, "db_engine"):
            db_args["engine"] = mimcRun.params.db_engine
        if hasattr(mimcRun.params, "db_name"):
            db_args["db"] = mimcRun.params.db_name
        db = mimcdb.MIMCDatabase(**db_args)
        run_id = db.createRun(mimc_run=mimcRun,
                              tag=mimcRun.params.db_tag)
        if fnItrDone is None:
            def ItrDone(db=db, r_id=run_id, r=mimcRun):
                    db.writeRunData(r_id, r, iteration_idx=len(r.iters)-1)
            fnItrDone = ItrDone
        else:
            fnItrDone = lambda db=db, r_id=run_id, r=mimcRun, fn=fnItrDone: \
                        fn(db, r_id, r)
    elif fnItrDone is not None:
        fnItrDone = lambda r=mimcRun, fn=fnItrDone: \
                        fn(None, None, r)
    mimcRun.setFunctions(fnItrDone=fnItrDone)

    try:
        mimcRun.doRun()
    except:
        if mimcRun.params.db and output_process():
            db.markRunFailed(run_id, totalTime=time.clock()-tStart)
        raise   # If you don't want to raise, make sure the following code is not executed

    if mimcRun.params.db and output_process():
        db.markRunSuccessful(run_id, totalTime=time.clock()-tStart)
    return mimcRun

def run_errors_est_program(fnExactErr=None):
    from . import db as mimcdb
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
                            help="Database Tags")
        parser.add_argument("-qoi_exact_tag", type=str, action="store")
        parser.add_argument("-qoi_exact", type=float, action="store",
                            help="Exact value")

    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    args = parse_known_args(parser)

    db_args = dict()
    if args.db_name is not None:
        db_args["db"] = args.db_name
    if args.db_user is not None:
        db_args["user"] = args.db_user
    if args.db_host is not None:
        db_args["host"] = args.db_host
    if args.db_engine is not None:
        db_args["engine"] = args.db_engine

    db = mimcdb.MIMCDatabase(**db_args)
    if args.db_tag is None:
        warnings.warn("You did not select a database tag!!")
    print("Reading data")

    run_data = db.readRuns(tag=args.db_tag)
    if len(run_data) == 0:
        raise Exception("No runs!!!")
    fnNorm = run_data[0].fn.Norm
    if args.qoi_exact_tag is not None:
        assert args.qoi_exact is None, "Multiple exact values given"
        exact_runs = db.readRuns(tag=args.qoi_exact_tag)
        from . import plot
        args.qoi_exact, _ = plot.estimate_exact(exact_runs)
        print("Estimated exact value is {}".format(args.qoi_exact))
        if fnExactErr is not None:
            fnExactErr = lambda itrs, r=exact_runs[0], fn=fnExactErr: fn(r, itrs)
    elif fnExactErr is not None:
        fnExactErr = lambda itrs, fn=fnExactErr: fn(itrs[0].parent, itrs)

    if args.qoi_exact is not None and fnExactErr is None:
        fnExactErr = lambda itrs, e=args.qoi_exact: \
                     fnNorm([v.calcEg() + e*-1 for v in itrs])

    print("Updating errors")
    db.update_exact_errors(run_data, fnExactErr)
