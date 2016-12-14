from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os.path
import numpy as np
import mimclib.test
import mimclib.miproj as miproj
from matern import SField_Matern
from mimclib import setutil
from mimclib import mimc, ipdb
import argparse

warnings.filterwarnings("error")
warnings.filterwarnings("always", category=mimclib.test.ArgumentWarning)

class MyRun:
    def solveFor_seq(self, alpha, arrY):
        output = np.zeros(len(arrY))
        self.sf.BeginRuns(alpha, np.max([len(Y) for Y in arrY]))
        for i, Y in enumerate(arrY):
            output[i] = self.sf.SolveFor(np.array(Y))
        self.sf.EndRuns()
        return output

    def mySampleQoI(self, run, lvls, M, moments):
        return self.proj.sample_all(run, lvls, M, moments,
                                    fnSample=self.solveFor_seq)

    def workModel(self, run, lvls):
        return mimc.work_estimate(lvls, np.log(run.params.beta) * run.params.gamma)

    def initRun(self, run):
        self.prev_val = 0
        self.qoi_problem = run.params.qoi_problem
        self.proj = miproj.MIWProjSampler(d=run.params.qoi_dim,
                                          fnBasis=miproj.legendre_polynomials,
                                          fnSamplePoints=miproj.sample_optimal_leg_pts,
                                          fnWorkModel=lambda lvls, r=run: self.workModel(run, lvls))
        self.proj.init_mimc_run(run)
        self.sf = SField_Matern(run.params)
        run.setFunctions(ExtendLvls=lambda lvls, r=run: self.extendLvls(run, lvls),
                         fnNorm=lambda arr: np.array([x.norm() for x in arr]))

        self.profit_calc = None
        if not run.params.qoi_set_adaptive:
            self.profit_calc = setutil.MIProfCalculator([run.params.qoi_set_dexp] * run.params.qoi_dim,
                                                        run.params.qoi_set_xi,
                                                        run.params.qoi_set_sexp)

    def extendLvls(self, run, lvls):
        if self.profit_calc is None:
            error = run.fn.Norm(run.last_itr.calcDeltaEl())
            work = run.last_itr.Wl_estimate
            prof = setutil.calc_log_prof_from_EW(error, work)
        else:
            prof = lvls.calc_log_prof(self.profit_calc)

        lvls.expand_set(prof,
                        seedLookahead=2, max_added=5,
                        profCalc=self.profit_calc)
        self.proj.update_index_set(lvls)

    def addExtraArguments(self, parser):
        class store_as_array(argparse._StoreAction):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, np.array(values))

        parser.add_argument("-qoi_dim", type=int, default=1, action="store")
        parser.add_argument("-qoi_problem", type=int, default=0, action="store")
        parser.add_argument("-qoi_a0", type=float, default=0., action="store")
        parser.add_argument("-qoi_f0", type=float, default=1., action="store")
        parser.add_argument("-qoi_df_nu", type=float, default=1., action="store")
        parser.add_argument("-qoi_df_L", type=float, default=1., action="store")
        parser.add_argument("-qoi_df_sig", type=float, default=1., action="store")
        parser.add_argument("-qoi_scale", type=float, default=1., action="store")
        parser.add_argument("-qoi_sigma", type=float, default=1., action="store")
        parser.add_argument("-qoi_x0", type=float, nargs='+',
                            default=np.array([0.4,0.2,0.6]),
                            action=store_as_array)

        parser.add_argument("-qoi_set_adaptive", type="bool",
                            default=True, action="store")
        parser.add_argument("-qoi_set_xi", type=float, default=2.,
                            action="store")
        parser.add_argument("-qoi_set_sexp", type=float, default=4.,
                            action="store")
        parser.add_argument("-qoi_set_dexp", type=float,
                            default=np.log(2.), action="store")


if __name__ == "__main__":
    SField_Matern.Init()
    from mimclib import ipdb
    ipdb.set_excepthook()

    run = MyRun()
    mirun = mimclib.test.RunStandardTest(fnSampleAll=run.mySampleQoI,
                                         fnAddExtraArgs=run.addExtraArguments,
                                         fnInit=run.initRun)
    SField_Matern.Final()
