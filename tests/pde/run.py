import numpy as np


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
    from pdelib.SField import SField
    SField.Init()
    with SField() as sf:
        import mimclib.test
        mimclib.test.RunStandardTest(fnSampleLvl=lambda *a: SamplePDE(sf, *a))

    SField.Final()
