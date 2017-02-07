from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os.path
import numpy as np

def addExtraArguments(parser):
    parser.add_argument("-qoi_K", type=float, default=0.4,
                        action="store")
    parser.add_argument("-qoi_sig", type=float, default=0.4,
                        action="store")
    parser.add_argument("-qoi_T", type=float, default=1.,
                        action="store")

import kuramoto

class ParticleField(object):
    def Init(self, run):
        self.gen = kuramoto.RandGen(run.params.qoi_seed)

    def SampleQoI(self, run, inds, M):
        meshes = (1./run.fn.Hierarchy(inds)).astype(np.int)
        import time
        tStart = time.clock()
        Ps = meshes[:, 0]
        Ns = meshes[:, 1] if run.params.min_dim == 2 \
                          else (4*meshes[:, 0]/run.params.h0inv[0]).astype(np.int)
        samples = kuramoto.SampleKuramoto(self.gen, Ns=Ns, Ps=Ps, M=M,
                                          T=run.params.qoi_T,
                                          K=run.params.qoi_K,
                                          sig=run.params.qoi_sig,
                                          var_sig=False,
                                          antithetic=True, dim=1)
        return samples, time.clock()-tStart

if __name__ == "__main__":
    import mimclib.test
    field = ParticleField()
    mimclib.test.RunStandardTest(fnInit=field.Init,
                                 fnSampleLvl=field.SampleQoI,
                                 fnAddExtraArgs=addExtraArguments)
