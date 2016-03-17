from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os.path
import numpy as np
import mimclib.mimc as mimc
import mimclib.db as mimcdb

warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                         "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                   lineno, cat.__name__, msg)
# warnings.filterwarnings('error')


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
    parser.add_argument("-qoi_sigma", type=float, default=1.,
                        action="store", help="Volatility in GBM")
    parser.add_argument("-qoi_mu", type=float, default=1.,
                        action="store", help="Drift in GBM")
    parser.add_argument("-qoi_T", type=float, default=1.,
                        action="store", help="Final time in GBM")
    parser.add_argument("-qoi_S0", type=float, default=1.,
                        action="store", help="Initial condition in GBM")
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

    mimcRun.setFunctions(fnSampleLvl=lambda *a: mySampleLvl(mimcRun, *a),
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

try:
    # Try to import the DLL version of wcumsum,
    # This makes solving the SDE much faster
    import ctypes as ct
    import numpy.ctypeslib as npct
    __arr_double__ = npct.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')
    __libdir = os.path.join(os.path.dirname(__file__))
    __lib__ = npct.load_library("_libwcumsum.so", __libdir)
    __lib__.wcumsum.restype = None
    __lib__.wcumsum.argtypes = [__arr_double__, __arr_double__,
                                ct.c_uint32, __arr_double__]

    def wcumsum(x, w):
        output = np.empty(len(x))
        __lib__.wcumsum(x, w, len(x), output)
        return output

except:
    warnings.warn("Using Python (very slow) version for wcumsum. Consider running make")
    # wcumsum is like cumsum, but weighted.
    def wcumsum(x, w):
        output = np.empty(len(x))
        output[0] = x[0]
        for i in range(1, len(output)):
            output[i] = w[i]*output[i-1] + x[i]
        return output


def mySampleLvl(run, moments, mods, inds, M):
    import time
    timeStart = time.time()
    psums = np.zeros(len(moments))
    meshes = (run.params.qoi_T/run.fnHierarchy(inds)).reshape(-1).astype(np.int)
    maxN = np.max(meshes)
    for m in range(0, M):
        dW = np.random.normal(size=maxN)/np.sqrt(maxN)
        solves = np.empty(len(mods))
        for i, mesh in enumerate(meshes):
            # Simple Code to solve SDE!
            assert(maxN % mesh == 0)
            dWl = np.sum(dW.reshape((-1, maxN//mesh)), axis=1)
            solves[i] = wcumsum(np.concatenate(([run.params.qoi_S0],
                                               np.zeros(len(dWl)))),
                                np.concatenate(([0],
                                               run.params.qoi_sigma*dWl +
                                               1 + run.params.qoi_mu/mesh)))[-1]
        psums += np.sum(mods*solves)**moments

    return M, psums, time.time() - timeStart

if __name__ == "__main__":
    main()
