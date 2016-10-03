from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os.path
import numpy as np
import mimclib.test
import mimclib.ipdb as ipdb
import mimclib.misc as misc
from matern import SField_Matern
from mimclib import setutil
from mimclib import mimc
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

    def mySampleQoI(self, run, inds, M):
        return self.misc.sample(inds, M, fnSample=self.solveFor_seq)

    def workModel(self, run, lvls):
        # from mimclib import ipdb
        # ipdb.embed()
        mat = lvls.to_dense_matrix()
        gamma = np.hstack((run.params.gamma, np.ones(mat.shape[1]-len(run.params.gamma))))
        beta = np.hstack((run.params.beta, 2*np.ones(mat.shape[1]-len(run.params.gamma))))
        return np.prod(beta**(mat*gamma), axis=1)

    def initRun(self, run):
        self.prev_val = 0
        self.extrapolate_s_dims = 10
        fnKnots= lambda beta: misc.knots_CC(misc.lev2knots_doubling(1+beta),
                                            -np.sqrt(3), np.sqrt(3))
        self.misc = misc.MISCSampler(d=run.params.min_dim, fnKnots=fnKnots)
        self.sf = SField_Matern(run.params)

        self.d_err_rates = 2.*np.log(run.params.beta) * \
                          np.minimum(1, run.params.qoi_df_nu / run.params.qoi_dim)
        self.d_work_rates = np.log(run.params.beta) * run.params.gamma
        # self.profCalc = setutil.MISCProfCalculator(self.d_err_rates +
        #                                            self.d_work_rates,
        #                                            np.ones(self.extrapolate_s_dims))

        run.setFunctions(ExtendLvls=lambda lvls, r=run: self.extendLvls(run, lvls),
                         WorkModel=lambda lvls, r=run: self.workModel(run, lvls))
        return


    def transNK(self, d, N, problem_arg=0):
        # return np.arange(0, N), np.arange(0, N)
        # Each ind has 2*|ind|_0 samples
        indSet = setutil.GenTDSet(d, N, base=0)
        N_per_ind = 2**np.sum(indSet!=0, axis=1)
        if problem_arg == 1:
            N_per_ind[1:] /= 2
        _, k_ind = np.unique(np.sum(indSet, axis=1), return_inverse=True)
        k_of_N = np.repeat(k_ind, N_per_ind.astype(np.int))[:N]
        # N_of_k = [j+np.arange(0, i, dtype=np.uint) for i, j in
        #           zip(N_per_ind, np.hstack((np.array([0],
        #                                              dtype=np.uint),
        #                                     np.cumsum(N_per_ind)[:np.max(k_of_N)])))]
        return k_of_N


    def extendLvls(self, run, lvls):
        if len(lvls) == 0:
            # First run, add min_lvls on each dimension
            d = run.params.min_dim + self.extrapolate_s_dims
            eye = np.eye(d, dtype=int)
            new_lvls = np.vstack([np.zeros(d, dtype=int)] +
                             [i*eye for i in range(1, run.params.min_lvls)])
            lvls.add_from_list(new_lvls)
            return

        # estimate rates
        self.d_err_rates, \
            s_fit_rates = misc.estimate_misc_error_rates(d=run.params.min_dim,
                                                         lvls=lvls,
                                                         errs=run.last_itr.calcDeltaEl(),
                                                         d_err_rates=self.d_err_rates,
                                                         lev2knots=lambda beta:misc.lev2knots_doubling(1+beta))
        #################### extrapolate error rates
        valid = np.nonzero(s_fit_rates > 1e-15)[0]  # rates that are negative or close to zero are not accurate.
        N = len(s_fit_rates) + self.extrapolate_s_dims
        k_of_N = self.transNK(run.params.qoi_dim, N, run.params.qoi_problem)
        K = np.max(k_of_N)

        c = np.polyfit(np.log(1+k_of_N[valid]), s_fit_rates[valid], 1)
        k_rates_stoch = c[0]*np.log(1+np.arange(0, K+1)) + c[1]
        s_err_rates = np.maximum(k_rates_stoch[k_of_N[:N]],
                                 np.min(s_fit_rates[valid]))
        s_err_rates[valid] = s_fit_rates[valid]  # The fitted rates should remain the same

        ######### Update
        self.profCalc = setutil.MISCProfCalculator(self.d_err_rates +
                                                   self.d_work_rates,
                                                   s_err_rates)
        mimc.extend_prof_lvls(lvls, self.profCalc, run.params.min_lvls)

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
                            default=[0.4,0.2,0.6], action=store_as_array)


if __name__ == "__main__":
    ipdb.set_excepthook()

    SField_Matern.Init()
    run = MyRun()
    mimclib.test.RunStandardTest(fnSampleLvl=run.mySampleQoI,
                                 fnAddExtraArgs=run.addExtraArguments,
                                 fnInit=run.initRun)
    SField_Matern.Final()
