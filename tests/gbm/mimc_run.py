"""
This file contains an example of a mimclib run.
The example solves a geometric Brownian motion SDE
with a given constant drift and volatility, with
given initial value and from zero to a given final value.

This file does NOT contain any parallel programming, but is
sequential.

In case of doubt, run the commands produced in echo_test_cmd.py.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
import time
from wcumsum import wcumsum

warnings.filterwarnings("error")


def addExtraArguments(parser):
    parser.add_argument("-qoi_sigma", type=float, default=1.,
                        action="store", help="Volatility in GBM")
    parser.add_argument("-qoi_mu", type=float, default=1.,
                        action="store", help="Drift in GBM")
    parser.add_argument("-qoi_T", type=float, default=1.,
                        action="store", help="Final time in GBM")
    parser.add_argument("-qoi_S0", type=float, default=1.,
                        action="store", help="Initial condition in GBM")
    parser.add_argument("-qoi_type", type=str, default="real",
                        action="store", help="Type of QoI. real, arr or obj")

import mimclib
class CustomClass(mimclib.mimc.custom_obj):
    def __init__(self, d):
        assert(np.isscalar(d))
        self.data = d

    def __add__(self, d):
        return CustomClass(self.data + d.data)

    def __mul__(self, scale):
        if isinstance(scale, CustomClass):
            return CustomClass(scale.data * self.data)
        return CustomClass(scale * self.data)

    def __pow__(self, power):
        return CustomClass(self.data**power)

    def __truediv__(self, scale):
        return CustomClass(self.data/scale)

    def __str__(self):
        return str(self.data)

    def __float__(self):
        return self.data

def mySampleQoI(run, inds, M):
    meshes = (run.params.qoi_T/run.fn.Hierarchy(inds)).reshape(-1).astype(np.int)
    maxN = np.max(meshes)

    tStart = time.clock()
    if run.params.qoi_type == "real":
        solves = np.empty((M, len(inds)), dtype=float)
    elif run.params.qoi_type == "obj":
        solves = np.empty((M, len(inds)), dtype=object)
    elif run.params.qoi_type == "arr":
        solves = np.empty((M, len(inds), 2), dtype=float)

    dW = np.random.normal(size=(M, maxN))/np.sqrt(maxN)
    for i, mesh in enumerate(meshes):
        assert(maxN % mesh == 0)
        dWl = np.sum(dW.reshape((M, -1, maxN//mesh)), axis=2)
        x = np.concatenate(([run.params.qoi_S0], np.zeros(dWl.shape[1])))
        w = np.zeros((dWl.shape[0], dWl.shape[1]+1))
        w[:, 1:] = run.params.qoi_sigma*dWl + 1 + run.params.qoi_mu/mesh
        val = wcumsum(x, w)[:, -1]
        if run.params.qoi_type == "real":
            solves[:, i] = val
        elif run.params.qoi_type == "obj":
            solves[:, i] = [CustomClass(d) for d in val]
        elif run.params.qoi_type == "arr":
            solves[:, i, 0] = val
            solves[:, i, 1] = val
    return solves, time.clock()-tStart

def initRun(run):
    if run.params.qoi_type == "obj":
        run.setFunctions(fnNorm=lambda x: np.array([np.abs(xx.data) for xx in x]))
    elif run.params.qoi_type == "arr":
        run.setFunctions(fnNorm=lambda x: np.max(np.abs(x), axis=1))
    elif run.params.qoi_type != "real":
        raise Exception("qoi_type option is not recognized")
    return


if __name__ == "__main__":
    import mimclib.test
    import sys
    mimclib.test.RunStandardTest(fnSampleLvl=mySampleQoI,
                                 fnAddExtraArgs=addExtraArguments,
                                 fnInit=initRun)
