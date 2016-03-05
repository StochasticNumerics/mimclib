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
warnings.filterwarnings('error')


def addExtraArguments(parser):
    def str2bool(v):
        # susendberg's function
        return v.lower() in ("yes", "true", "t", "1")
    parser.register('type', 'bool', str2bool)
    parser.add_argument("-db_user", type=str, default=None,
                        action="store", help="Database User")
    parser.add_argument("-db_host", type=str, default='localhost',
                        action="store", help="Database Host")
    parser.add_argument("-db_tag", type=str,
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


def MLMCPDE(DB=True):
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    mimc.MIMCRun.addOptionsToParser(parser)
    mimcRun = mimc.MIMCRun(**vars(parser.parse_known_args()[0]))
    if mimcRun.params.qoi_seed >= 0:
        np.random.seed(mimcRun.params.qoi_seed)

    fnItrDone = None
    if mimcRun.params.db:
        db = mimcdb.MIMCDatabase(user=mimcRun.params.db_user,
                                 host=mimcRun.params.db_host)
        run_id = db.createRun(mimc_run=mimcRun, tag="NoTag")
        fnItrDone = lambda *a: db.writeRunData(run_id, mimcRun, *a)

    mimcRun.setFunctions(fnSampleLvl=lambda *a: mySampleLvl(mimcRun, *a),
                         fnItrDone=fnItrDone)

    mimcRun.doRun()
    return mimcRun.data.calcEg()

try:
    import ctypes as ct
    import numpy.ctypeslib as npct
    __arr_double__ = npct.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')
    __libdir = os.path.join(os.path.dirname(__file__))
    __lib__ = npct.load_library("_libwcumsum.so", __libdir)
    __lib__.wcumsum.restype = None
    __lib__.wcumsum.argtypes = [__arr_double__, __arr_double__,
                                ct.c_uint32, ct.c_double, __arr_double__]

    def wcumsum(x, w, x0=0):
        output = np.empty(len(x))
        __lib__.wcumsum(x, w, len(x), x0, output)
        return output

except:
    raise
    warnings.warn("Using Python (very slow) version for wcumsum. Consider running make")
    def wcumsum(x, w, x0=0):
        output = np.empty(len(x))
        output[0] = w[0]*x0 + x[0]
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
            dWl = np.sum(dW.reshape((-1, maxN//mesh)), axis=1)
            solves[i] = wcumsum(run.params.qoi_sigma*dWl,
                                np.ones(len(dWl)) +
                                run.params.qoi_mu/mesh,
                                x0=run.params.qoi_S0)[-1]
        psums += np.sum(mods*solves)**moments

    return psums, time.time() - timeStart

if __name__ == "__main__":
    MLMCPDE()
    #TestDB()
