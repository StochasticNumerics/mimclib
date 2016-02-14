import warnings
import os.path
import numpy as np
import mimclib.mimc as mimc
import mimclib.db as mimcdb

warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                         "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                   lineno, cat.__name__, msg)
warnings.filterwarnings('error')


def TestDB():
    db = mimcdb.MIMCDatabase(user="abdo")
    data = db.readRunData(db.getRunDataIDs())
    import IPython
    IPython.embed()


def MLMCPDE(DB=True):
    import argparse
    from pdelib.SField import SField
    SField.Init()

    parser = argparse.ArgumentParser(add_help=True)
    mimc.MIMCRun.addOptionsToParser(parser)
    sf = SField()
    if DB:
        db = mimcdb.MIMCDatabase(user="abdo")

    mimcRun = mimc.MIMCRun(**vars(parser.parse_known_args()[0]))
    if DB:
        run_id = db.createRun(mimc_run=mimcRun, tag="NoTag")
    mimcRun.setFunctions(fnSampleLvl=lambda *a: SamplePDE(sf, mimcRun, *a),
                         fnItrDone=lambda *a: db.writeRunData(run_id, mimcRun,
                                                              *a))

    mimcRun.doRun()
    SField.Final()
    return mimcRun.data.calcEg()


def SamplePDE(sf, run, moments, mods, inds, M):
    import time
    timeStart = time.time()
    psums = np.zeros(len(moments))
    sf.BeginRuns(mods, 1./run.fnHierarchy(inds))
    for m in range(0, M):
        psums += sf.Sample()**moments
    sf.EndRuns()
    return psums, time.time() - timeStart

if __name__ == "__main__":
    MLMCPDE()
    #TestDB()
