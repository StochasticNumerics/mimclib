from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os.path
import numpy as np
import HJM 

def addExtraArguments(parser):
    parser.add_argument("-qoi_sigma", type=float, default=1.,
                        action="store", help="Volatility in GBM")
    parser.add_argument("-qoi_mu", type=float, default=1.,
                        action="store", help="Drift in GBM")
    parser.add_argument("-qoi_T", type=float, default=1.,
                        action="store", help="Final time in GBM")
    parser.add_argument("-qoi_S0", type=float, default=1.,
                        action="store", help="Initial condition in GBM")

def wrapFunMIMC(inds):
    offset = 1
    inp = [[foo[0]+offset, foo[1]+offset, foo[1]+offset] for foo in inds]
    return HJM.infDimTest(inp)

def wrapFunMLMC(inds):
    offset = 1
    return HJM.infDimTest([[foo[0]+offset]*3 for foo in inds])

def mySampleQoI(run, inds):
    return wrapFunMIMC(inds)

if __name__ == "__main__":
    import mimclib.test
    mimclib.test.RunStandardTest(fnSampleQoI=mySampleQoI,
                                 fnAddExtraArgs=addExtraArguments)
