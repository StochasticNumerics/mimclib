from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import warnings
import os.path
import numpy as np
import mimclib.test
import mimclib.miproj as miproj
from mimclib import setutil
from mimclib import mimc, ipdb
import argparse

warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=mimclib.test.ArgumentWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class MyRun:
    def solveFor_sin(self, alpha, arrY):
        j = np.arange(0, arrY.shape[1], dtype=np.float)
        output = np.sin(np.sum(1. / (1+((1.+j[None, :]) **
                                        (-(self.params.qoi_df_nu+0.5)) *
                                        arrY)**2.), axis=1))
        return output

    def solveFor_sf(self, alpha, arrY):
        output = np.zeros(len(arrY))
        self.sf.BeginRuns(alpha, np.max([len(Y) for Y in arrY]))
        for i, Y in enumerate(arrY):
            output[i] = self.sf.SolveFor(np.array(Y))
        self.sf.EndRuns()
        return output

    def solveFor_kl1D(self, alpha, arrY):
        from kl1D import kl1D
        assert(len(alpha) == 1)
        return kl1D(arrY, 2**alpha[0], self.params.qoi_df_nu + 0.5)[:, 0]

    def solveFor_matern(self, alpha, arrY):
        from matern_fem import matern
        assert(len(alpha) == 1)
        return matern(arrY, 2**alpha[0],
                      nu=self.params.qoi_df_nu,
                      df_sig=self.params.qoi_df_sig,
                      df_L=self.params.qoi_df_L,
                      qoi_x0=self.params.qoi_x0[0],
                      qoi_sig=self.params.qoi_sigma)[:, 0]

    def solveFor_seq(self, alpha, arrY):
        if len(alpha) == 0:
            alpha = [self.params.miproj_fix_lvl] * self.params.qoi_dim
        if self.params.qoi_problem == 'matern':
            return self.solveFor_sf(alpha, arrY)
        if self.params.qoi_problem == 'matern-py':
            return self.solveFor_matern(alpha, arrY)
        if self.params.qoi_problem == 'kl1D':
            return self.solveFor_kl1D(alpha, arrY)
        if self.params.qoi_problem == 'sin':
            return self.solveFor_sin(alpha, arrY)

    def mySampleQoI(self, run, lvls, M, moments):
        return self.proj.sample_all(run, lvls, M, moments,
                                    fnSample=self.solveFor_seq)

    def initRun(self, run):
        self.prev_val = 0
        self.params = run.params

        if run.params.miproj_pts_sampler == 'optimal':
            fnSamplePoints = miproj.sample_optimal_leg_pts
            fnWeightPoints = lambda x, b: miproj.optimal_weights(b)
        elif run.params.miproj_pts_sampler == 'arcsine':
            fnSamplePoints = miproj.sample_arcsine_pts
            fnWeightPoints = lambda x, b: miproj.arcsine_weights(x)
        else:
            raise NotImplementedError("Unknown points sampler")

        if run.params.min_dim > 0:
            fnWorkModel = lambda lvls, w=0.5*np.log(run.params.beta) * run.params.gamma: \
                          mimc.work_estimate(lvls, w)
        else:
            fnWorkModel = lambda lvls, w=run.params.beta ** (run.params.gamma/2.):\
                          w * np.ones(len(lvls))

        self.proj = miproj.MIWProjSampler(d=run.params.min_dim,
                                          min_dim=np.minimum(run.params.qoi_min_vars, run.params.miproj_max_dim),
                                          fnBasis=miproj.legendre_polynomials,
                                          fnBasisFromLvl=miproj.default_basis_from_level,
                                          fnSamplePoints=fnSamplePoints,
                                          fnWeightPoints=fnWeightPoints,
                                          fnWorkModel=fnWorkModel,
                                          reuse_samples=run.params.miproj_reuse_samples)
        self.proj.init_mimc_run(run)
        if self.params.qoi_problem == 'matern':
            from matern import SField_Matern
            SField_Matern.Init()
            self.sf = SField_Matern(run.params)
        run.setFunctions(ExtendLvls=lambda lvls, r=run: self.extendLvls(run, lvls),
                         fnNorm=lambda arr: np.array([x.norm() for x in arr]))

        self.profit_calc = None
        if not run.params.qoi_set_adaptive:
            self.profit_calc = setutil.MIProfCalculator([run.params.qoi_set_dexp] * run.params.min_dim,
                                                        run.params.qoi_set_xi,
                                                        run.params.qoi_set_sexp,
                                                        run.params.qoi_set_mul)
            # self.profit_calc = setutil.MISCProfCalculator([run.params.qoi_set_dexp] * run.params.min_dim,
            #                                               [run.params.qoi_set_sexp] * run.params.miproj_max_dim)

    def extendLvls(self, run, lvls):
        max_added = None
        max_dim = 5 + (0 if len(lvls) == 0 else np.max(lvls.get_dim()))
        max_dim = np.minimum(run.params.miproj_max_dim,
                             np.maximum(run.params.miproj_min_dim, max_dim))
        tStart = time.clock()
        if self.profit_calc is None:
            # Adaptive
            error = run.fn.Norm(run.last_itr.calcDeltaEl())
            work = run.last_itr.Wl_estimate
            prof = setutil.calc_log_prof_from_EW(error, work)
            max_added = 30
            lvls.expand_set(prof, max_dim=max_dim, max_added=max_added)
            self.proj.update_index_set(lvls)
        else:
            # non-adaptive
            prof = self.profit_calc
            prev_total_work = self.proj.estimateWork()
            while True:
                lvls.expand_set(prof, max_dim=max_dim, max_added=max_added)
                self.proj.update_index_set(lvls)
                new_total_work = self.proj.estimateWork()
                if not self.params.qoi_double_work or new_total_work >= 2*prev_total_work:
                    break
        if self.params.verbose >= 1:
            print("Time taken to extend levels: ", time.clock()-tStart)

    def addExtraArguments(self, parser):
        class store_as_array(argparse._StoreAction):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, np.array(values))

        parser.add_argument("-qoi_dim", type=int, default=1, action="store")
        parser.add_argument("-qoi_problem", type=str, default="matern", action="store")
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
        parser.add_argument("-qoi_min_vars", type=int,
                            default=10, action="store")
        parser.add_argument("-qoi_double_work", type="bool",
                            default=False, action="store")

        parser.add_argument("-qoi_set_adaptive", type="bool",
                            default=True, action="store")
        parser.add_argument("-qoi_set_xi", type=float, default=2.,
                            action="store")
        parser.add_argument("-qoi_set_mul", type=float, default=1.,
                            action="store")
        parser.add_argument("-qoi_set_sexp", type=float, default=4.,
                            action="store")
        parser.add_argument("-qoi_set_dexp", type=float,
                            default=np.log(2.), action="store")

        parser.add_argument("-miproj_pts_sampler", type=str,
                            default="optimal", action="store")
        parser.add_argument("-miproj_reuse_samples", type="bool",
                            default=True, action="store")
        parser.add_argument("-miproj_fix_lvl", type=int,
                            default=3, action="store")
        parser.add_argument("-miproj_min_dim", type=int,
                            default=2, action="store")
        parser.add_argument("-miproj_max_dim", type=int,
                            default=10**6, action="store")

    def ItrDone(self, db, run_id, run):
        if db is not None:
            db.writeRunData(run_id, run,
                            iteration_idx=len(run.iters)-1,
                            userdata=self.proj.user_data)
        self.proj.user_data = []

if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()

    run = MyRun()
    mirun = mimclib.test.RunStandardTest(fnSampleAll=run.mySampleQoI,
                                         fnAddExtraArgs=run.addExtraArguments,
                                         fnInit=run.initRun,
                                         fnItrDone=run.ItrDone)
    if mirun.params.qoi_problem == 'matern':
        from matern import SField_Matern
        SField_Matern.Final()
