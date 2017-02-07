from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os.path
import numpy as np
import mimclib.test
import mimclib.misc as misc
from mimclib import setutil
from mimclib import mimc
import argparse

warnings.filterwarnings("error")
warnings.filterwarnings("always", category=mimclib.test.ArgumentWarning)

class MyRun:
    def func_oscillatory(self, run, alpha, arrY):
        arrY = np.array(arrY)
        return np.cos(2*np.pi*run.params.qoi_w[0] + np.sum(run.params.qoi_c * arrY, axis=1))

    def func_productpeak(self, run, alpha, arrY):
        arrY = np.array(arrY)
        return np.prod((run.params.qoi_c*-2. + (arrY - run.params.qoi_w)**2.)**-1, axis=1)

    def func_cornerpeak(self, run, alpha, arrY):
        arrY = np.array(arrY)
        return (1 + np.sum(run.params.qoi_c*arrY, axis=1)) ** (-(run.params.qoi_dim+1))

    def func_gauss(self, run, alpha, arrY):
        arrY = np.array(arrY)
        return np.exp(-np.sum(run.params.qoi_c**2 *
                              (arrY - run.params.qoi_w)**2, axis=1))

    def func_cont(self, run, alpha, arrY):
        arrY = np.array(arrY)
        return np.exp(-np.sum(run.params.qoi_c * np.abs(arrY - run.params.qoi_w)))

    def func_discont(self, run, alpha, arrY):
        raise NotImplementedError()

    def mySampleQoI(self, inds, M):
        return self.misc.sample(inds, M, fnSample=self.func)

    def workModel(self, run, lvls):
        mat = lvls.to_dense_matrix()
        return np.prod(2**mat , axis=1) * 2.**np.sum(mat > 0, axis=1)

    def initRun(self, run):
        if run.params.qoi_knots.lower() == "uniform":
            fnKnots = lambda beta: misc.knots_uniform(misc.lev2knots_linear(1+beta),
                                                      0, 1, 'nonprob')
        else:
            assert(run.params.qoi_knots.lower() == "cc")
            fnKnots = lambda beta: misc.knots_CC(misc.lev2knots_doubling(1+beta),
                                                 0, 1, 'nonprob')

        self.misc = misc.MISCSampler(d=0, fnKnots=fnKnots, min_dim=run.params.min_dim)

        b = [9.0, 7.25, 1.85, 7.03, 2.04, 4.3]
        fn = [self.func_oscillatory, self.func_productpeak,
              self.func_cornerpeak, self.func_gauss, self.func_cont,
              self.func_discont]
        self.func = lambda inds, M, r=run: fn[run.params.qoi_func-1](r, inds, M)

        np.random.seed(run.params.qoi_seed)
        run.params.qoi_w = 1./np.arange(1.,run.params.qoi_dim+1, dtype=np.float)#np.random.random(size=run.params.qoi_dim)
        run.params.qoi_c = np.arange(1,run.params.qoi_dim+1, dtype=np.float) #np.random.random(size=run.params.qoi_dim)
        run.params.qoi_c *= b[run.params.qoi_func-1] / np.sum(run.params.qoi_c)

        run.params.qoi_c = np.array([2.71,0.16,3.42,2.71])
        run.params.qoi_w = np.array([0.42,0.72,0.01,0.3])

        run.setFunctions(ExtendLvls=lambda lvls, r=run: self.extendLvls(run, lvls),
                         WorkModel=lambda lvls, r=run: self.workModel(run, lvls),
                         fnSampleLvl=self.mySampleQoI)
        self.profCalc = setutil.TDProfCalculator(np.ones(run.params.qoi_dim))
        return

    def extendLvls(self, run, lvls):
        if len(lvls) == 0:
            # First run, add min_lvls on each dimension
            d = run.params.min_dim
            eye = np.eye(d, dtype=int)
            new_lvls = np.vstack([np.zeros(d, dtype=int)] +
                             [i*eye for i in range(1, run.params.min_lvls)])
            lvls.add_from_list(new_lvls)
            return
        mimc.extend_prof_lvls(lvls, self.profCalc, run.params.min_lvls)
        return

        # import time
        # tStart = time.clock()
        # # estimate rates
        # self.d_err_rates, \
        #     s_fit_rates = misc.estimate_misc_error_rates(d=0,
        #                                                  lvls=lvls,
        #                                                  errs=run.last_itr.calcDeltaEl(),
        #                                                  d_err_rates=np.zeros([]),
        #                                                  lev2knots=lambda beta:misc.lev2knots_doubling(1+beta))
        # #################### extrapolate error rates
        # tEnd_rates = time.clock() - tStart
        # ######### Update
        # tStart = time.clock()
        # self.profCalc = setutil.MISCProfCalculator(np.zeros(0), s_fit_rates)
        # mimc.extend_prof_lvls(lvls, self.profCalc, run.params.min_lvls)

    def addExtraArguments(self, parser):
        class store_as_array(argparse._StoreAction):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, np.array(values))

        parser.add_argument("-qoi_dim", type=int, default=10, action="store")
        parser.add_argument("-qoi_knots", type=str, default="cc", action="store")
        parser.add_argument("-qoi_func", type=int, default=1, action="store")


if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()

    run = MyRun()
    mimclib.test.RunStandardTest(fnSampleLvl=run.mySampleQoI,
                                 fnAddExtraArgs=run.addExtraArguments,
                                 fnInit=run.initRun)
