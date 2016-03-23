from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os.path
import numpy as np
import mimclib.mimc as mimc
import mimclib.db as mimcdb

import HJM

warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                         "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                   lineno, cat.__name__, msg)
#warnings.filterwarnings('error')


def addExtraArguments(parser):
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

def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    mimc.MIMCRun.addOptionsToParser(parser)
    mimcRun = mimc.MIMCRun(**vars(parser.parse_known_args()[0]))
    if mimcRun.params.qoi_seed >= 0:
        np.random.seed(mimcRun.params.qoi_seed)

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

    mimcRun.setFunctions(fnSampleQoI=lambda *a: mySampleQoI(mimcRun, *a),
                         fnItrDone=fnItrDone)

    try:
        mimcRun.doRun()
        if mimcRun.params.db:
            # The run succeeded, mark it as done in the database
            db.markRunDone(run_id)
    except:
        # The run failed, mark it as failed in the database
        if mimcRun.params.db:
            db.markRunDone(run_id, flag=0)
        raise

    return mimcRun.data.calcEg()

def mySampleQoI(run, inds):
    return HJM.hoLeeExample(inds)

if __name__ == "__main__":
    main()
