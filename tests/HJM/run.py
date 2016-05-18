from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import HJM
import warnings
import os.path
import numpy as np

def python_wcumsum(x, w):
    assert(len(x) == w.shape[1])
    output = np.empty_like(w)
    for m in range(0, w.shape[0]):
        output[m, 0] = x[0]
        for i in range(1, w.shape[1]):
            output[m, i] = w[m, i]*output[m, i-1] + x[i]
    return output

def addExtraArguments(parser):
    parser.add_argument("-qoi_sigma", type=float, default=1.,
                        action="store", help="Volatility in GBM")
    parser.add_argument("-qoi_mu", type=float, default=1.,
                        action="store", help="Drift in GBM")
    parser.add_argument("-qoi_T", type=float, default=1.,
                        action="store", help="Final time in GBM")
    parser.add_argument("-qoi_S0", type=float, default=1.,
                        action="store", help="Initial condition in GBM")

def mySampleQoI(run, inds, M):
    tStart = time.time()
    return np.array([testcude(inds) for m in range(M)]), time.time()-tStart

if __name__ == "__main__":
    import mimclib.test
    mimclib.test.RunStandardTest(fnSampleLvl=mySampleQoI,
                                 fnAddExtraArgs=addExtraArguments)
