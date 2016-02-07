from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import itertools
import warnings
import set_util


class MIMCData(object):
    def __init__(self, dim):
        self.lvls = []                            # MIMC lvls
        self.psums = np.empty(0, 2)               # sums of lvls
        self.t = np.array(list())                 # Time of lvls
        self.M = np.array(list(), dtype=np.int)   # Number of samples in each lvl

    def calcEg(self):
        return np.sum(self.calcEl())

    def Dim(self):
        return self.dim

    def calcVl(self):
        return self.psums[:, 1] / self.M - (self.calcEl())**2

    def calcEl(self):
        return self.psums[:, 0] / self.M

    def calcTl(self):
        return self.t / self.M

    def calcTotalTime(self):
        return np.sum(self.Tl() * self.M)

    def addSamples(self, psums, M, t):
        self.psums += psums
        self.M += M
        self.t += t

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
provided!".format(key))


class MIMCRun(object):
    def __init__(self, **kwargs):
        self.data = MIMCData(dim=kwargs["dim"])
        self.params = MyDefaultDict(kwargs)
        self.params.set_defaults(bayesian=False, absBnd=False,
                                 reuse_samples=True,
                                 const_theta=False)
        self.params.set_warn_defaults(Ca=3, theta=0.5,
                                      fnWorkModel=lambda x: x.Tl())

        self.bias = np.inf           # Approximation of the discretization error
        self.stat_error = np.inf     # Sampling error (based on M)

        if self.bayesian and 'fnWorkModel' not in kwargs:
            raise NotImplementedError("Bayesian parameter fitting is only \
supported with a given work model")

        if self.bayesian and self.dim > 1:
            raise NotImplementedError("Bayesian parameter fitting is only \
supported in one dimensional problem")

    def calcTotalWork(self):
        return np.sum(self.fnWorkModel() * self.data.M)

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

    def estimateVl(self):
        if not self.params.baeysian:
            return self.data.calcVl()
        # TODO: Implement bayesian estimation of Vl

    def estimateBias(self):
        if not self.params.baeysian:
            bnd = calcBoundary(self.lvls)   # TODO: Call function from set_util
            if np.sum(bnd) == len(self.lvls):
                return np.inf
            bnd_val = self.data.calcEl()[bnd]
            if self.params.absBnd:
                return np.abs(np.sum(np.abs(bnd_val)))
            return np.abs(np.sum(bnd_val))
        # TODO: Implement bayesian estimation of bias

    def estimateParams(self):
        if not self.params.baeysian:
            return
        # TODO: Estimate Q_S, Q_W, q_1, q_2

    def estimateOptimalL(self):
        assert(self.params.baeysian)
        # TODO: Estimate optimal L

    def calcSamples(self, fnSamplelvl, lvls, totalM, verbose):
        s = len(lvls)
        M = np.zeros(s, dtype=np.int)
        psums = np.zeros((s, 2))
        p = np.arange(1, psums.shape[1])
        t = np.zeros(s)
        for i in range(0, s):
            if totalM[i] <= self.data.M[i]:
                continue
            if verbose:
                print("# Doing", totalM[i]-self.data.M[i], "of level", lvls[i])
            inds, mods = lvl_to_inds_general(lvls[i])
            psums[i, :], t[i] = fnSamplelvl(self, p, mods, inds,
                                            totalM[i] - self.data.M[i])
            M[i] = totalM[i]
        return psums, M, t

    def doRun(self, finalTOL, TOLs, fnExtendLvls, fnSampleLvls, fnItrDone=None, verbose=False):
        # fnExtendLvls, fnSamplelvl
        # fnExtendLvls(MIMCRun): Returns new lvls and number of samples on each.
        #                        called only once if the Bayesian method is used
        # fnSampleLvls(MIMCRun, moments, mods, inds, M):
        #    Returns array: M sums of mods*inds, and total (linear) time it took to compute them
        # fnItrDone(MIMCRun, i, TOLs): Called at the end of iteration i out of TOLs
        if len(self.data.lvls) != 0:
            warnings.warn("Running the same object twice, resetting")
            self.data.reset()
        if not all(x >= y for x, y in zip(TOLs, TOLs[1:])):
            raise Exception("Tolerances must be decreasing")

        import time
        tic = time.time()
        newLvls, todoM = fnExtendLvls(self)
        self.data.addLevels(newLvls)
        self.calcSamples(fnSampleLvls, todoM, verbose)

        def calcTheoryM(tol, Vl, Wl, theta):
            return (theta * tol / self.params.Ca)**-2 * np.sum(np.sqrt(Wl * Vl)) * np.sqrt(Vl / Wl)

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
                        self.data.addLevels([[i] for i in
                                             range(len(self.data.lvls), L+1)])
                theta = -1
                if not self.params.const_theta:
                    theta = 1 - self.estimateBias() / TOL
                if theta <= 0:
                    theta = self.params.theta   # Bias too large or const_theta
                if verbose:
                    print("# theta", theta)
                todoM = np.int_(calcTheoryM(TOL, self.estimateVl(),
                                            self.fnWorkModel(self), theta))
                if verbose:
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
                    self.calcSamples(fnSampleLvls, todoM, verbose)
            if fnItrDone:
                fnItrDone(self, itrIndex, TOL)
            if TOL < finalTOL:
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

def work_estimate(beta, gamma, lvls):
    return np.prod(beta**(np.array(lvls, dtype=np.int) * np.array(gamma)), axis=1)

def boundary_gen(d, lvls):
    # TODO: Use set_util
    bnd = np.zeros(len(lvls), dtype=int)
    for i in range(0, d):
        x = np.zeros(d)
        x[i] = 1
        bnd += np.array([1 if l[i] == 0 or (np.array(l)+x).tolist() in lvls else 0 for l in lvls])
    return np.nonzero(bnd < d)[0]


def lvl_to_inds_general(lvl):
    # TODO: Use set_util
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
