from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
import sys
import mimclib.test
# import IPython.ultraTB
# sys.excepthook = IPython.ultraTB.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=mimclib.test.ArgumentWarning)

def SamplePDE(sf, run, inds, M):
    import time
    timeStart = time.time()
    solves = np.zeros((M, len(inds)))
    sf.BeginRuns(1./run.fn.Hierarchy(inds))
    for m in range(0, M):
        solves[m, :] = sf.Sample()
    sf.EndRuns()
    return solves, time.time() - timeStart

if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()

    from pdelib.SField import SField
    SField.Init()
    with SField() as sf:
        mimclib.test.RunStandardTest(fnSampleLvl=lambda *a: SamplePDE(sf, *a))
    SField.Final()
