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
    parser.register('type', 'bool',
                    lambda v: v.lower() in ("yes", "true", "t", "1"))
    parser.add_argument("-db_user", type=str,
                        action="store", help="Database User")
    parser.add_argument("-db_host", type=str, default='localhost',
                        action="store", help="Database Host")
    parser.add_argument("-db_tag", type=str, default="NoTag",
                        action="store", help="Database Tag")
    parser.add_argument("-db", type='bool', default=False,
                        action="store", help="Save in Database")
    parser.add_argument("-qoi_seed", type=int, default=-1,
                        action="store", help="Seed for random generator")


def main():
    import argparse
    from pdelib.SField import SField
    SField.Init()

    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    mimc.MIMCRun.addOptionsToParser(parser)
    sf = SField()

    mimcRun = mimc.MIMCRun(**vars(parser.parse_known_args()[0]))
    if mimcRun.params.qoi_seed >= 0:
        np.random.seed(mimcRun.params.qoi_seed)
    fnItrDone = None
    if mimcRun.params.db:
        if hasattr(mimcRun.params, "db_user"):
            db = mimcdb.MIMCDatabase(user=mimcRun.params.db_user,
                                     host=mimcRun.params.db_host)
        else:
            db = mimcdb.MIMCDatabase(host=mimcRun.params.db_host)
        run_id = db.createRun(mimc_run=mimcRun,
                              tag=mimcRun.params.db_tag)
        fnItrDone = lambda *a: db.writeRunData(run_id, mimcRun, *a)

    mimcRun.setFunctions(fnSampleLvl=lambda *a: SamplePDE(sf, mimcRun, *a),
                         fnItrDone=fnItrDone)

    try:
        mimcRun.doRun()
    except:
        if mimcRun.params.db:
            db.markRunFailed(run_id)
        raise   # If you don't want to raise, make sure the following code is not executed

    SField.Final()
    if mimcRun.params.db:
            db.markRunSuccessful(run_id)
    return mimcRun.data.calcEg()


def SamplePDE(sf, run, moments, mods, inds, M):
    import time
    timeStart = time.time()
    psums = np.zeros(len(moments))
    sf.BeginRuns(mods, 1./run.fnHierarchy(inds))
    for m in range(0, M):
        psums += sf.Sample()**moments
    sf.EndRuns()
    return M, psums, time.time() - timeStart

if __name__ == "__main__":
    main()
