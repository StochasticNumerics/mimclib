from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#warnings.filterwarnings('error')

import numpy as np
def RunStandardTest(fnSampleQoI=None, fnSampleLvl=None,
                    fnAddExtraArgs=None, fnSeed=np.random.seed):
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
    parser.add_argument("-db_host", type=str, default='localhost',
                        action="store", help="Database Host")
    parser.add_argument("-db_tag", type=str, default="NoTag",
                        action="store", help="Database Tag")
    parser.add_argument("-db", type='bool', default=False,
                        action="store", help="Save in Database")
    parser.add_argument("-qoi_seed", type=int, default=-1,
                        action="store", help="Seed for random generator")

    if fnAddExtraArgs is not None:
        fnAddExtraArgs(parser)
    mimc.MIMCRun.addOptionsToParser(parser)
    mimcRun = mimc.MIMCRun(**vars(parser.parse_known_args()[0]))
    if mimcRun.params.qoi_seed >= 0 and fnSeed is not None:
        fnSeed(mimcRun.params.qoi_seed)

    fnItrDone = None
    if mimcRun.params.db:
        if hasattr(mimcRun.params, "db_user"):
            db = mimcdb.MIMCDatabase(user=mimcRun.params.db_user,
                                     host=mimcRun.params.db_host)
        else:
            db = mimcdb.MIMCDatabase(host=mimcRun.params.db_host)
        run_id = db.createRun(mimc_run=mimcRun,
                              tag=mimcRun.params.db_tag)
        fnItrDone = lambda *a: db.writeRunData(run_id, mimcRun, *a)

    if fnSampleQoI is not None:
        fnSampleQoI = lambda inds, fn=fnSampleQoI: fn(mimcRun, inds)
    if fnSampleLvl is not None:
        fnSampleLvl = lambda moments, mods, inds, M, fn=fnSampleLvl, *a: fn(mimcRun, moments, mods, inds, M)

    mimcRun.setFunctions(fnSampleQoI=fnSampleQoI,
                         fnSampleLvl=fnSampleLvl, fnItrDone=fnItrDone)

    try:
        mimcRun.doRun()
    except:
        if mimcRun.params.db:
            db.markRunFailed(run_id)
        raise   # If you don't want to raise, make sure the following code is not executed

    if mimcRun.params.db:
        db.markRunSuccessful(run_id)
    return mimcRun.data.calcEg()
