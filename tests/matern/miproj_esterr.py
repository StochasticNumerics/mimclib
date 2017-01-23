#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mimclib import ipdb
import warnings
import time

warnings.filterwarnings("always")

def l2_error_mc(itrs, fnSample, rel_tol=0.1, maxM=2000):
    if len(itrs) == 0:
        return np.array([])
    max_L = 1 + np.max([np.max([a[0] for a in itr.lvls_itr()]) for itr in itrs])
    N = 200
    try:
        N = itrs[0].parent.params.qoi_N
    except:
        pass
    if N < 0:
        N = np.max([itr.parent.last_itr.lvls_max_dim()-1 for itr in itrs])

    tStart = time.time()
    print("MaxL:", max_L, ", N:", N)
    # Evaluate L^2 norm using Monte Carlo
    val = [itr.calcEg() for itr in itrs]
    s1 = np.zeros(len(itrs))
    s2 = np.zeros(len(itrs))
    M = 0
    nextM = 10
    np.random.seed(0)
    while M < maxM:
        Y = np.random.uniform(-1, 1, size=(nextM-M, N))
        samples = fnSample(itr.parent, [max_L], Y)
        errors = np.zeros((nextM-M, len(itrs)))
        for i in xrange(0, len(itrs)):
            errors[:, i] = (samples - val[i](Y))**2
        s1 += np.sum(errors, axis=0)
        s2 += np.sum(errors**2, axis=0)
        M += len(samples)
        err = 3*np.sqrt((s2/M - (s1/M)**2)/M)    # Approximate error of error estimate
        max_rel_err = np.max(np.sqrt(err / (s1/M)))
        print("Estimate with", M, "samples -> max relative error: ", max_rel_err,
              "in", (time.time()-tStart)/60., "min")
        if max_rel_err < rel_tol:
            break
        nextM *= 2
    return np.sqrt(s1/M)

if __name__ == "__main__":
    from miproj_run import MyRun
    sampler = MyRun()

    from mimclib import ipdb
    ipdb.set_excepthook()
    from mimclib import test
    test.run_errors_est_program(lambda iters:
                                l2_error_mc(iters, sampler.solveFor_seq))
