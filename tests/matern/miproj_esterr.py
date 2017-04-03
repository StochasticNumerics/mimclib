#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mimclib import ipdb
import warnings
import time

warnings.filterwarnings("always")

def l2_error_mc(itrs, fnSample, rel_tol=0.01, maxM=1000, max_L=10):
    if len(itrs) == 0:
        return np.array([])

    if max_L is None:
        max_L = np.max([
            itr.parent.params.miproj_fix_lvl
            if itr.parent.params.min_dim == 0
            else 1+np.max([a[0] for a in itr.lvls_itr()]) for itr in itrs])

    N = -1
    try:
        N = np.max([itr.parent.params.qoi_N for itr in itrs])
    except:
        pass

    if N < 0:
        N = np.max([itr.parent.last_itr.lvls_max_dim()-itr.parent.params.min_dim for itr in itrs])
        N += 10

    try:
        N = np.minimum(N, np.max([itr.parent.params.miproj_max_dim for itr in itrs]))
    except:
        pass

    tStart = time.clock()
    print("MaxL:", max_L, ", N:", N)
    # Evaluate L^2 norm using Monte Carlo
    val = [itr.calcEg() for itr in itrs]
    s1 = np.zeros(len(itrs))
    s2 = np.zeros(len(itrs))

    s1_E = 0
    s2_E = 0
    M = 0
    nextM = 10
    np.random.seed(0)
    while M < maxM:
        Y = np.random.uniform(-1, 1, size=(nextM-M, N))
        samples = fnSample([max_L], Y)
        errors = np.zeros((nextM-M, len(itrs)))
        for i in xrange(0, len(itrs)):
            errors[:, i] = (samples - val[i](Y))**2
        s1 += np.sum(errors, axis=0)
        s2 += np.sum(errors**2, axis=0)
        s1_E += np.sum(samples)
        s2_E += np.sum(samples**2)

        M += len(samples)
        err = 3*np.sqrt((s2/M - (s1/M)**2)/M)    # Approximate error of error estimate
        max_rel_err = np.max(err / (err+np.abs(s1/M)))
        print("Estimate with", M, "samples -> max relative error: ", max_rel_err,
              "in", (time.clock()-tStart)/60., "min")
        print("Expectation:", s1_E/M, ", Error:", 3*np.sqrt((s2_E/M - (s1_E/M)**2)/M))
        if max_rel_err < rel_tol:
            break
        nextM *= 2
    return np.sqrt(s1/M)

if __name__ == "__main__":
    import miproj_run
    sampler = miproj_run.MyRun()
    sampler.params = None
    def fnSample(run, iters):
        if sampler.params is None:
            sampler.params = run.params
            if sampler.params.qoi_problem == 'matern':
                from matern import SField_Matern
                SField_Matern.Init()
                sampler.sf = SField_Matern(sampler.params)
        return l2_error_mc(iters, sampler.solveFor_seq)

    from mimclib import ipdb
    ipdb.set_excepthook()
    from mimclib import test
    test.run_errors_est_program(fnSample)
    if sampler.params.qoi_problem == 'matern':
        from matern import SField_Matern
        SField_Matern.Final()
