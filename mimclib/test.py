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
                    fnAddExtraArgs=None,
                    fnInit=None,
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
    parser.add_argument("-qoi_seed", type=int, default=-1,
                        action="store", help="Seed for random generator")
    parser.add_argument("-db_name", type=str, action="store", help="")

    if fnAddExtraArgs is not None:
        fnAddExtraArgs(parser)
    mimc.MIMCRun.addOptionsToParser(parser)
    mimcRun = mimc.MIMCRun(**vars(parse_known_args(parser)))
    fnSampleLvl = lambda inds, M, fn=fnSampleLvl: fn(mimcRun, inds, M)
    mimcRun.setFunctions(fnSampleLvl=fnSampleLvl)

    import time
    tStart = time.time()
    if fnInit is not None:
        fnInit(mimcRun)

    if mimcRun.params.qoi_seed >= 0 and fnSeed is not None:
        fnSeed(mimcRun.params.qoi_seed)

    fnItrDone = None

    if mimcRun.params.db:
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
        fnItrDone = lambda: db.writeRunData(run_id,
                                            mimcRun,
                                            iteration_idx=len(mimcRun.iters)-1)

    mimcRun.setFunctions(fnItrDone=fnItrDone)

    try:
        mimcRun.doRun()
    except:
        if mimcRun.params.db:
            db.markRunFailed(run_id, totalTime=time.time()-tStart)
        raise   # If you don't want to raise, make sure the following code is not executed

    if mimcRun.params.db:
        db.markRunSuccessful(run_id, totalTime=time.time()-tStart)
    return mimcRun.last_itr.calcEg()
