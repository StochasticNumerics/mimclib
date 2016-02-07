from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import itertools
import warnings
import set_util


class MIMCData(object):
    def __init__(self, dim, psums=np.empty(0, 2), t=np.empty(0),
                 M=np.empty(0, dtype=np.int)):
        self.dim = dim
        self.lvls = []                            # MIMC lvls
        self.psums = psums               # sums of lvls
        self.t = t                 # Time of lvls
        self.M = M   # Number of samples in each lvl

    def calcEg(self):
        return np.sum(self.calcEl())

    def __getitem__(self, ind):
        return MIMCData(self.dim, psums=self.psums[ind, :],
                        t=self.t[ind], M=self.M[ind])

    def Dim(self):
        return self.dim

    def calcVl(self):
        return self.psums[:, 1] / self.M - (self.calcEl())**2

    def calcEl(self, moment=1):
        assert(moment>0)
        return self.psums[:, moment-1] / self.M

    def calcTl(self):
        return self.t / self.M

    def calcTotalTime(self, ind=None):
        return np.sum(self.Tl() * self.M)

    def addSamples(self, psums, M, t):
        assert psums.shape[0] == len(M) and len(M) == len(t), \
            "Inconsistent arguments "

        self.psums += psums
        self.M += M
        self.t += t

    def zero_samples(self):
        self.M = 0
        self.t = 0
        self.psums = 0

    def addLevels(self, new_lvls):
        assert(len(new_lvls) > 0)
        prev = len(self.lvls)
        self.lvls.extend(new_lvls)
        s = len(self.lvls)
        self.psums.resize((s, self.psums.shape[1]), refcheck=False)
        self.t.resize(s, refcheck=False)
        self.M.resize(s, refcheck=False)
        return prev


class MyDefaultDict(object):
    def __init__(self, kwargs):
        self.__dict__ = kwargs.copy()
        self.__defaults__ = dict()
        self.__warn_defaults__ = dict()

    def set_defaults(self, **kwargs):
        self.__defaults__ = kwargs

    def set_warn_defaults(self, **kwargs):
        self.__warn_defaults__ = kwargs

    def __getattr__(self, name):
        if name in self.__defaults__:
            return self.__defaults__[name]
        if name in self.__warn_defaults__:
            default_val = self.__warn_defaults__[name]
            warnings.warn("Argument '{}' is required but not provided,\
default value '{}' is used.".format(name, default_val))
            return default_val
        raise NameError("Argument '{}' is required but not \
provided!".format(name))


class MIMCRun(object):
    def __init__(self, **kwargs):
        self.params = MyDefaultDict(kwargs)
        self.params.set_defaults(bayesian=False, absBnd=False,
                                 reuse_samples=True,
                                 const_theta=False)
        self.params.set_warn_defaults(Ca=3, theta=0.5,
                                      fnWorkModel=lambda x: x.Tl())

        self.bias = np.inf           # Approximation of the discretization error
        self.stat_error = np.inf     # Sampling error (based on M)

        self.data = MIMCData(dim=self.params.dim)
        self.all_data = MIMCData(dim=self.params.dim)
        # If self.params.reuse_samples is True then
        # all_data will always equal data

        if self.params.bayesian and 'fnWorkModel' not in kwargs:
            raise NotImplementedError("Bayesian parameter fitting is only \
supported with a given work model")

        if self.params.bayesian and 'fnHierarchy' not in kwargs:
            raise NotImplementedError("Bayesian parameter fitting is only \
supported with a given hierarchy")

        if self.params.bayesian and self.dim > 1:
            raise NotImplementedError("Bayesian parameter fitting is only \
supported in one dimensional problem")

        if self.params.bayesian:
            self.Q = MyDefaultDict(S=0, W=0, w=self.params.w,
                                   s=self.params.s)

    def calcTotalWork(self):
        return np.sum(self.fnWorkModel(self, self.lvls) * self.data.M)

    def estimateStatError(self):
        return self.params.Ca * \
            np.sqrt(np.sum(self.data.estimateVl() / self.data.M))

    def estimateTotalError(self):
        return self.estimateBias() + self.estimateStateError()

    def __str__(self):
        output = "Time={:.12e}\nEg={:.12e}\n\
\Bias={:.12e}\nstatErr={:.12e}".format(self.data.calcTotalTime(),
                                       self.data.calcEg(),
                                       self.bias,
                                       self.stat_error)
        V = self.estimateVl()
        E = self.data.calcEl()
        T = self.data.calcTl()

        output += ("{:<8}{:^20}{:^20}{:>8}{:>10}{:>15}{:>8}".format(
            "Level", "E", "V", "M", "theoryM", "Time", "Var%"))
        for i in range(0, len(self.lvls)):
            output += ("{:<8}{:>+20.12e}{:>20.12e}{:>8}{:>10.2f}{:>15.6e}{:>8.2f}%".format(
                self.lvls[i], E[i], V[i], self.data.M[i], self.theoryM[i], T[i],
                100 * np.sqrt(V[i]) / np.abs(E[i])))
        return output

    def _estimateVl(self):
        if not self.params.baeysian:
            return self.data.calcVl()
        return self._estimateBaysianVl()

    ################## Bayesian specific functions
    def estimateBias(self):
        if not self.params.baeysian:
            bnd = is_boundary(self.data.dim, self.lvls)
            if np.sum(bnd) == len(self.lvls):
                return np.inf
            bnd_val = self.data[bnd].calcEl()
            if self.params.absBnd:
                return np.abs(np.sum(np.abs(bnd_val)))
            return np.abs(np.sum(bnd_val))
        return _estimateBayesianBias(self)

    def _estimateBayesianBias(self, L=None):
        L = L or 1+len(self.all_data.lvls)
        hl = self.params.fnHierarchy(self, np.arange(0,L+1).reshape((-1,1)))
        return np.abs(self.Q.W) * hl[-1]**self.Q.w

    def _estimateBayesianVl(self, L=None):
        L = L or 1+len(self.all_data.lvls)
        # TODO: need to correct this code for larger L values
        hl = self.params.fnHierarchy(self, np.arange(0,L+1).reshape((-1,1)))
        M = self.all_data[1:].M
        m1 = self.all_data[1:].calcEl()
        m2 = self.all_data[1:].calcEl(moment=2)
        mu = self.Q.W*(hl[1:]**self.Q.w - hl[:-1]**self.Q.w)
        Lambda = 1./(self.Q.S*(hl[1:]**(self.Q.s/2.) - hl[:-1]**(self.Q.s/2.))**2)
        G_3 = self.params.kappa1 * Lambda + M
        G_4 = self.params.kappa1 + \
              0.5*M*(m2-m1**2 + self.kappa0 * (m1 - mu)**2 /
                     (self.kappa0 + M))
        return np.concatenate(self.data[0].calcVl(), G_4 / G_3)

    def _estimateParams(self):
        if not self.params.baeysian:
            return
        hl = self.params.fnHierarchy(self)
        begin = 1
        M = self.all_data[begin:].M
        m1 = self.all_data[begin:].calcEl()
        m2 = self.all_data[begin:].calcEl(moment=2)
        wl = hl[begin:]**self.Q.w - hl[(begin-1):-1]**self.Q.w
        sl = (hl[begin:]**(self.Q.s/2.) - hl[(begin-1):-1]**(self.Q.s/2.))**-2

        self.Q.W = np.sum(wl * sl * M * m1) / np.sum(M * wl**2 * sl)
        self.Q.S = np.sum(sl * (m2 - 2*m1*self.Q.W*wl + self.Q.W**2*wl**2)) / np.sum(M)
        if self.params.w_sig > 0 or self.params.s_sig > 0:
            # TODO: Estimate w=q_1, s=q_2
            raise NotImplemented("TODO, estimate w and s")

    def _estimateOptimalL(self):
        assert self.params.baeysian, "MIMC should be Bayesian to \
estimate optimal number of levels"
        minL = len(self.lvls)
        minWork = np.inf
        for L in range(len(self.lvls), len(self.lvls)+1+self.params.incL):
            Wl = self.params.fnWorkModel(self,
                                         np.arange(0, L+1).reshape((-1, 1)))
            M, _ = self._calcTheoryM(TOL,
                                     bias_est=self._estimateBayesianBias(L),
                                     self._estimateBayesianVl(L), Wl)
            totalWork = np.sum(Wl*M)
            if totalWork < minWork:
                minL = L
                minWork = totalWork
        return minL
    ################## END: Bayesian specific function

    def _addSamples(self, psums, M, t):
        self.data.addSamples(psums, M, t)
        self.all_data.addSamples(psums, M, t)

    def _calcSamples(self, fnSamplelvl, lvls, totalM, verbose):
        s = len(lvls)
        M = np.zeros(s, dtype=np.int)
        psums = np.zeros((s, 2))
        p = np.arange(1, psums.shape[1])
        t = np.zeros(s)
        for i in range(0, s):
            if totalM[i] <= self.data[i].M:
                continue
            if verbose:
                print("# Doing", totalM[i]-self.data[i].M, "of level", lvls[i])
            inds, mods = lvl_to_inds_general(lvls[i])
            psums[i, :], t[i] = fnSamplelvl(self, p, mods, inds,
                                            totalM[i] - self.data[i].M)
            M[i] = totalM[i]
        return psums, M, t

    def _calcTheoryM(self, TOL, bias_est, Vl, Wl):
        theta = -1
        if not self.params.const_theta:
            theta = 1 - bias_est/TOL
        if theta <= 0:
            theta = self.params.theta   # Bias too large or const_theta
        return (theta * TOL / self.params.Ca)**-2 *\
            np.sum(np.sqrt(Wl * Vl)) * np.sqrt(Vl / Wl), theta

    def doRun(self, finalTOL, TOLs, fnExtendLvls, fnSampleLvls, fnItrDone=None, verbose=False):
        # fnExtendLvls, fnSamplelvl
        # fnExtendLvls(MIMCRun): Returns new lvls and number of samples on each.
        #                        called only once if the Bayesian method is used
        # fnSampleLvls(MIMCRun, moments, mods, inds, M):
        #    Returns array: M sums of mods*inds, and total (linear) time it took to compute them
        # fnItrDone(MIMCRun, i, TOLs): Called at the end of iteration i out of TOLs
        # fnWorkModel(MIMCRun, lvls): Returns work estimate of lvls
        # fnHierarchy(MIMCRun, lvls): Returns associated hierarchy of lvls
        if len(self.data.lvls) != 0:
            warnings.warn("Running the same object twice, resetting")
            self.data = MIMCData(self.data.dim)

        if not all(x >= y for x, y in zip(TOLs, TOLs[1:])):
            raise Exception("Tolerances must be decreasing")

        import time
        tic = time.time()
        newLvls, todoM = fnExtendLvls(self)
        self.addLevels(newLvls)
        self.calcSamples(fnSampleLvls, todoM, verbose)

        import gc
        for itrIndex, TOL in enumerate(TOLs):
            if verbose:
                print("# TOL", TOL)
            while True:
                gc.collect()
                self.estimateParams()
                if self.params.bayesian:
                    L = self.optimalL()
                    if L > len(self.data.lvls):
                        self.data._addLevels(np.arange(len(self.data.lvls),
                                                       L+1).reshape((-1, 1)))
                todoM, theta = self._calcTheoryM(TOL,
                                                 self.estimateBias(),
                                                 self._estimateVl(),
                                                 self.fnWorkModel(self,
                                                                  self.lvls))
                todoM = np.int_(todoM)
                if verbose:
                    print("# theta", theta)
                    print("# New M: ", todoM)
                if not self.params.reuse_samples:
                    self.data.zero_samples()
                self.calcSamples(fnSampleLvls, todoM, verbose)
                self.stat_error = self.estimateStatError()
                self.bias = self.estimateBias()
                self.totalTime = time.time() - tic
                if verbose:
                    self.textOutput()
                    print("------------------------------------------------")
                if self.params.bayesian or (self.bias + self.stat_error < TOL):
                    if verbose:
                        print("{} took {}".format(TOL, self.totalTime))
                    break
                if self.bias > (1 - theta) * TOL:
                    # Bias is not satisfied. Add more levels
                    newlvls, todoM = fnExtendLvls(self)
                    prev = len(self.lvls)
                    self.addLevels(newlvls)
                    todoM = np.zeros(len(self.lvls))
                    todoM[prev:] = todoM
                    self._calcSamples(fnSampleLvls, todoM, verbose)
            if fnItrDone:
                fnItrDone(self, itrIndex, TOL)
            if TOL <= finalTOL:
                break

def lvls_tensor(run):
    d, lvls = run.data.dim, run.data.lvls
    if len(lvls) <= 0:
        return [[0] * d]
    deg = np.max([np.max(ll) for ll in lvls])
    newlvls = list()
    additions = [f for f in itertools.product([0, 1], repeat=d) if max(f) > 0]
    for l in [ll for ll in lvls if np.max(ll) == deg]:
        newlvls.extend([(np.array(l) + a).tolist() for a in additions
                        if np.max(np.array(l) + a) == deg + 1 and (np.array(l) + a).tolist() not in newlvls])
    return newlvls


def lvls_td(run, w):
    lvls = run.data.lvls
    prev_deg = np.max(np.sum(np.array(
        [w*np.array(l) for l in lvls]), axis=1)) if lvls else 0
    max_deg = prev_deg
    while True:
        max_deg += np.min(w)
        C, _ = set_util.AnisoProfCalculator(w*0, w).GetIndexSet(max_deg)
        all_lvls = C.to_dense_matrix() - 1
        newlvls = [lvl.tolist() for lvl in all_lvls if lvl.tolist() not in lvls]
        if len(newlvls) > 0:
            return newlvls


def work_estimate(hl, gamma):
    return hl**gamma


def is_boundary(d, lvls):
    bnd = np.zeros(len(lvls), dtype=int)
    for i in range(0, d):
        x = np.zeros(d)
        x[i] = 1
        bnd += np.array([1 if l[i] == 0 or (np.array(l)+x).tolist() in lvls else 0 for l in lvls])
    return bnd < d


def lvl_to_inds_general(lvl):
    lvl = np.array(lvl, dtype=np.int)
    seeds = list()
    for i in range(0, lvl.shape[0]):
        if lvl[i] == 0:
            seeds.append([0])
        else:
            seeds.append([0, 1])
    inds = np.array(list(itertools.product(*seeds)), dtype=np.int)
    mods = (2 * np.sum(lvl) % 2 - 1) * (2 * (np.sum(inds, axis=1) % 2) - 1)
    return mods, np.tile(lvl, (inds.shape[0], 1)) - inds


def get_geometric_hl(lvls, beta):
    return np.prod(beta**(np.array(lvls, dtype=np.int)), axis=1)


def get_tol_sequence(TOL, maxTOL, max_additional_itr=1, r1=2, r2=1.1):
    # number of iterations until TOL
    eni = np.int(-(np.log(TOL)-np.log(maxTOL))/np.log(r1))
    adjTOL = TOL/r2
    return np.concatenate((TOL*r1**np.arange(eni, -1, -1),
                           TOL*r2**np.arange(1, max_additional_itr)))


def get_optimal_hl(mimc):
    if mimc.data.dim != 1:
        raise NotImplemented("Optimized hierarchies are only supported\
 for one-dimensional problems")

    # TODO: Get formula from HajiAli 2015, Optimizing MLMC hierarchies
    raise NotImplemented("TODO: get_optimal_hl")
