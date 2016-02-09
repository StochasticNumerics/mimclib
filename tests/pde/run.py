import numpy as np
import site
import mimc

import warnings
import os.path
warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                         "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                   lineno, cat.__name__, msg)
warnings.filterwarnings('error')

from pdelib.SField import SField

def main():
    import argparse
    SField.Init()

    parser = argparse.ArgumentParser(add_help=True)
    mimc.MIMCRun.addOptionsToParser(parser)
    sf = SField()
    mimcRun = mimc.MIMCRun(fnWorkModel=lambda r, lvls:
                           mimc.work_estimate(lvls, r.params.gamma),
                           fnHierarchy=lambda r, lvls:
                           mimc.get_geometric_hl(lvls, r.params.h0inv,
                                                 r.params.beta),
                           **vars(parser.parse_known_args()[0]))

    mimcRun.doRun(fnSampleLvls=lambda *a: Sample_PDE(sf, *a))

    SField.Final()

def Sample_PDE(sf, run, moments, mods, inds, M):
    import time
    timeStart = time.time()
    psums = np.zeros(len(moments))
    sf.BeginRuns(mods, 1./run.params.fnHierarchy(run, inds))
    for m in range(0, M):
        sample = sf.Sample()
        psums += sample**moments
    sf.EndRuns()
    return psums, time.time() - timeStart

if __name__ == "__main__":
    main()
