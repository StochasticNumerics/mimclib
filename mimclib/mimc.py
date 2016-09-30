from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import itertools
import warnings
from . import setutil

__all__ = []
import argparse

def public(sym):
    __all__.append(sym.__name__)
    return sym

@public
class custom_obj(object):
    # Used to add samples. Supposedly not used if M is fixed to 1
    def __add__(self, d):  # d type is custom_obj
        raise NotImplementedError("You should implement the __add__ function")

    # Used to compute delta samples. Supposedly only multiples 1 and -1
    # Might be more if the combination technique is used.
    def __mul__(self, scale): # scale is float
        raise NotImplementedError("You should implement the __mul__ function")

    # Used for moment computation. Supposedly not used if moments==1
    def __pow__(self, power): # power is float
        raise NotImplementedError("You should implement the __mul__ function")

    # Used to compute moments from sums. Supposedly not used if M is fixed to 1
    def __truediv__(self, scale): # scale is integer
        if scale == 1:
            return self
        raise NotImplementedError("You should implement the __truediv__ function")

    def __sub__(self, d):
        return self + d*-1

class _empty_obj(object):
    def __add__(self, newarr):
        return newarr  # Forget about this object

@public
def compute_raw_moments(psums, M):
    '''
    Returns the raw moments or None when M=0.
    '''
    idx = M != 0
    val = np.empty_like(psums)
    val[idx] = psums[idx] / _expand(M[idx], 0, val[idx].shape)
    #np.tile(M[idx].reshape((-1,1)), (1, psums.shape[1]))
    val[M == 0, :] = None
    return val

@public
def compute_central_moment(psums, M, moment):
    '''
    Returns the centralized moments or None when M=0.
    '''
    raw = compute_raw_moments(psums, M)
    if moment == 1:
        return raw[:, 0]

    n = moment
    pn = np.array([n])
    val = (raw[:, 0]**pn) * (-1)**n
    # From http://mathworld.wolfram.com/CentralMoment.html
    nfact = np.math.factorial(n)
    for k in range(1, moment+1):
        nchoosek = nfact / (np.math.factorial(k) * np.math.factorial(n-k))
        val +=  (raw[:, k-1] * raw[:, 0]**(pn-k)) * nchoosek * (-1)**(n-k)

    if moment % 2 == 1:
        return val
    return val
    # The moment should be positive
    # TODO: Debug for objects
    if np.min(val)<0.0:
        """
        There might be kurtosis values that are actually
        zero but slightly negative, smaller in magnitude
        than the machine precision. Fixing these manually.
        """
        idx = np.abs(val) < np.finfo(float).eps
        val[idx] = np.abs(val[idx])
        if np.min(val)<0.0:
            warnings.warn("Significantly negative {}'th moment = {}! \
Possible problem in computing sums.".format(moment, np.min(val)))
    return val

def _expand(b, i, shape):
    assert(len(b) == shape[i])
    b_shape = np.ones(len(shape), dtype=np.int)
    b_shape[i] = len(b)
    b_reps = list(shape)
    b_reps[i] = 1
    return np.tile(b.reshape(b_shape), b_reps)

@public
class MIMCData(object):
    """
    MIMC Data is a class for describing necessary data
    for a MIMC data, such as the dimension of the problem,
    list of levels, times exerted, sample sizes, etc...

    In a MIMC Run object, the data is stored in a MIMCData object

    """

    def __init__(self, min_dim=0, lvls=None, psums_delta=None,
                 psums_fine=None, t=None, M=None, moments=2):
        self.moments = moments
        import copy
        if lvls is not None:
            self.lvls = lvls
        else:
            self.lvls = setutil.VarSizeList(min_dim=min_dim)

        self.psums_delta = None
        self.psums_fine = None
        self.t = np.zeros(len(self.lvls))      # Time of lvls
        self.M = np.zeros(len(self.lvls), dtype=np.int)      # Number of samples in each lvl

        if psums_delta is not None:
            self.psums_delta = psums_delta.copy()
        if psums_fine is not None:
            self.psums_fine = psums_fine.copy()

        if t is not None:
            self.t = t.copy()
        if M is not None:
            self.M = M.copy()

        if self.psums_fine is not None:
            assert(len(self.lvls) == len(self.psums_fine))
        if self.psums_delta is not None:
            assert(len(self.lvls) == len(self.psums_delta))
        assert(len(self.lvls) == len(self.M))
        assert(len(self.lvls) == len(self.t))

    def calcEg(self):
        """
        Return the sum of the sample estimators for
        all the levels
        """
        return np.sum(self.calcDeltaEl(), axis=0)

    def __len__(self):
        return len(self.lvls)

    def __getitem__(self, ind):
        # Reshape everything to allow ind to be a single index or a list
        return MIMCData(lvls=self.lvls.sublist(ind),
                        psums_delta=self.psums_delta[ind].reshape((-1,) + self.psums_delta.shape[1:]),
                        psums_fine=self.psums_fine[ind].reshape((-1,) + self.psums_delta.shape[1:]),
                        t=self.t[ind].reshape(-1),
                        M=self.M[ind].reshape(-1))

    def computedMoments(self):
        return self.moments

    def calcDeltaVl(self):
        if self.moments < 2:
            vl = np.empty(len(self))
            vl.fill(np.nan)
            return vl
        return self.calcDeltaCentralMoment(2)

    def calcDeltaEl(self, moment=1):
        '''
        Returns the sample estimators for moments
        for each level.
        '''
        if moment > self.psums_delta.shape[1]:
            raise ValueError("The {}'th moment was not computed".format(moment))
        assert(moment > 0)
        idx = self.M != 0
        val = np.empty_like(self.psums_delta[:, moment-1])
        val[idx] = self.psums_delta[idx, moment-1] / \
                   _expand(self.M[idx], 0 ,self.psums_delta[idx, moment-1].shape)
        val[np.logical_not(idx)] = None
        return val

    def calcDeltaCentralMoment(self, moment):
        return compute_central_moment(self.psums_delta, self.M, moment)

    def calcFineCentralMoment(self, moment):
        return compute_central_moment(self.psums_delta, self.M, moment)

    def calcTl(self):
        idx = self.M != 0
        val = np.zeros_like(self.M, dtype=np.float)
        val[idx] = self.t[idx] / self.M[idx]
        return val

    def calcTotalTime(self, ind=None):
        return np.sum(self.t, axis=0)

    def addSamples(self, lvl_idx, M, psums_delta, psums_fine, t):
        assert psums_delta.shape == psums_fine.shape and \
            psums_fine.shape[0] == self.computedMoments(), "Inconsistent arguments "
        if self.M[lvl_idx] == 0:
            if self.psums_delta is None:
                self.psums_delta = np.zeros((len(self.lvls),)
                                            + psums_delta.shape, dtype=psums_delta.dtype)

            if self.psums_fine is None:
                self.psums_fine = np.zeros((len(self.lvls),)
                                            + psums_fine.shape, dtype=psums_fine.dtype)

            self.psums_delta[lvl_idx] = psums_delta
            self.psums_fine[lvl_idx] = psums_fine
            self.M[lvl_idx] = M
            self.t[lvl_idx] = t
        else:
            self.psums_delta[lvl_idx] += psums_delta
            self.psums_fine[lvl_idx] += psums_fine
            self.M[lvl_idx] += M
            self.t[lvl_idx] += t
        if psums_delta.dtype != self.psums_delta.dtype:
            self.psums_delta = self.psums_delta.astype(psums_delta.dtype)
        if psums_fine.dtype != self.psums_fine.dtype:
            self.psums_fine = self.psums_fine.astype(psums_fine.dtype)

    def zero_samples(self):
        self.M = np.zeros_like(self.M)
        self.t = np.zeros_like(self.t)
        self.psums_delta = np.empty((len(lvls), moments), dtype=object)
        self.psums_fine = np.empty((len(lvls), moments), dtype=object)

    def _levels_added(self):
        prev = len(self.M)
        s = len(self.lvls)
        if s == prev:
            return
        if self.psums_delta is not None:
            self.psums_delta.resize((s, ) + self.psums_delta.shape[1:], refcheck=False)
        if self.psums_fine is not None:
            self.psums_fine.resize((s, ) + self.psums_fine.shape[1:], refcheck=False)

        self.t.resize(s, refcheck=False)
        self.M.resize(s, refcheck=False)
        return prev


class MyDefaultDict(object):
    def __init__(self, **kwargs):
        self.__dict__ = dict([i for i in kwargs.items() if i[1] is not None])

    def getDict(self):
        return self.__dict__

    def __getattr__(self, name):
        raise AttributeError("Argument '{}' is required but not \
provided!".format(name))


@public
class MIMCRun(object):

    """
    Object for a Multi-Index Monte Carlo run.

    Data levels, moment estimators, sample sizes etc. are
    stored in the *.data attribute that is of the MIMCData type

    """

    def __init__(self, old_data=None, **kwargs):
        self.fn = MyDefaultDict(# Hierarchy=None, ExtendLvls=None,
                                # WorkModel=None, SampleLvl=None,
                                # ItrDone=None,
                                Norm=np.abs)
        self.params = MyDefaultDict(**kwargs)
        self.Vl_estimate = None
        self.Wl_estimate = None
        self.bias = np.inf           # Approximation of the discretization error
        self.stat_error = np.inf     # Sampling error (based on M)
        if old_data is not None:
            self.all_data = self.data = old_data
        else:
            self.all_data = self.data = MIMCData(min_dim=self.params.min_dim,
                                                 moments=self.params.moments)
            if not self.params.reuse_samples:
                self.all_data = MIMCData(lvls=self.data.lvls,
                                         min_dim=self.params.min_dim,
                                         moments=self.params.moments)

        dims = np.array([len(getattr(self.params, a))
                for a in ["w", "s", "gamma", "beta"] if hasattr(self.params, a)])
        if len(dims) > 0 and np.any(dims != dims[0]):
            raise ValueError("Size of beta, w, s and gamma must be of size dim")

        if self.params.bayesian:
            self.Q = MyDefaultDict(S=np.inf, W=np.inf,
                                   w=self.params.w, s=self.params.s,
                                   theta=np.nan)
        else:
            self.Q = MyDefaultDict(theta=np.nan)

    def _checkFunctions(self):
        # If self.params.reuse_samples is True then
        # all_data will always equal data
        if not hasattr(self.fn, "WorkModel") and hasattr(self.params, "gamma"):
            self.fn.WorkModel = lambda lvls: work_estimate(lvls,
                                                          np.log(self.params.beta) *
                                                          self.params.gamma)

        if not hasattr(self.fn, "Hierarchy"):
            self.fn.Hierarchy = lambda lvls: get_geometric_hl(lvls,
                                                             self.params.h0inv,
                                                             self.params.beta)

        if self.params.bayesian and not hasattr(self.fn, "WorkModel"):
            raise NotImplementedError("Bayesian parameter fitting is only \
supported with a given work model")

        if not hasattr(self.fn, "WorkModel"):
            # ADDING WORK MODEL B
            warnings.warn("fnWorkModel is not provided, using run-time estimates.")
            raise NotImplemented("Need to check that the lvls \
are the same as the argument ones")
            self.fn.WorkModel = lambda lvls: self.Tl()

        if self.fn.SampleLvl is None:
            raise ValueError("Must set the sampling functions fnSampleLvl")

        if not hasattr(self.fn, "ExtendLvls"):
            weights = self.params.beta * (self.params.w +
                                          (self.params.s -
                                           self.params.gamma)/2.)
            weights /= np.sum(weights, axis=0)
            profCalc = setutil.TDProfCalculator(weights)
            self.fn.ExtendLvls = lambda lvls: extend_prof_lvls(lvls, profCalc,
                                                               self.params.min_lvls)

    def setFunctions(self, **kwargs):
        # fnSampleLvl(moments, mods, inds, M):
        #    Returns M, array: M sums of mods*inds, and total
        #    (linear) time it took to compute them
        # fnItrDone(i, TOLs, totalTime): Called at the end of iteration
        #    i out of TOLs
        # fnWorkModel(lvls): Returns work estimate of lvls
        # fnHierarchy(lvls): Returns associated hierarchy of lvls
        for k in kwargs.keys():
            kk = k[2:] if k.startswith('fn') else k
            if kk not in ["SampleLvl", "ExtendLvls",
                         "ItrDone", "WorkModel",
                         "Hierarchy", "SampleQoI", "Norm"]:
                raise KeyError("Invalid function name")
            setattr(self.fn, kk, kwargs[k])

    @staticmethod
    def addOptionsToParser(parser, pre='-mimc_', additional=True, default_bayes=True):
        def str2bool(v):
            # susendberg's function
            return v.lower() in ("yes", "true", "t", "1")
        mimcgrp = parser.add_argument_group('MIMC', 'Arguments to control MIMC logic')
        mimcgrp.register('type', 'bool', str2bool)

        class Store_as_array(argparse._StoreAction):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, np.array(values))

        def add_store(name, action="store", **kwargs):
            if "default" in kwargs and "help" in kwargs:
                kwargs["help"] += " (default: {})".format(kwargs["default"])
            mimcgrp.add_argument(pre + name, dest=name,
                                 action=action,
                                 **kwargs)

        add_store('min_dim', type=int, default=0, help="Number of minimum dimensions used in the index set.")
        add_store('verbose', type='bool', default=False,
                  help="Verbose output")
        add_store('bayesian', type='bool', default=False,
                  help="Use Bayesian fitting to estimate bias, variance and optimize number \
of levels in every iteration. This is based on CMLMC.")
        add_store('moments', type=int, default=4, help="Number of moments to compute")
        add_store('reuse_samples', type='bool', default=True,
                  help="Reuse samples between iterations")
        add_store('abs_bnd', type='bool', default=False,
                  help="Take absolute value of deltas when \
estimating bias (sometimes that's too conservative).")
        add_store('const_theta', type='bool', default=False,
                  help="Use the same theta for all iterations")
        add_store('confidence', type=float, default=0.95,
                  help="Parameter to control confidence level")
        add_store('theta', type=float, default=0.5,
                  help="Minimum theta or error splitting parameter.")
        add_store('incL', type=int, default=2,
                  help="Maximum increment of number of levels \
between iterations")
        add_store('w', nargs='+', type=float, action=Store_as_array,
                  help="Weak convergence rates. \
Not needed if a profit calculator is specified and -bayesian is False.")
        add_store('s', nargs='+', type=float, action=Store_as_array,
                  help="Strong convergence rates.  \
Not needed if a profit calculator is specified and -bayesian is False.")
        add_store('TOL', type=float,
                  help="The required tolerance for the MIMC run")
        add_store('beta', type=float, nargs='+', action=Store_as_array,
                  help="Level separation parameter. to be used \
with get_geometric_hl. Not needed if fnHierarchy is provided.")
        add_store('gamma', type=float, nargs='+', action=Store_as_array,
                  help="Work exponent to be used with work_estimate.\
Not needed if fnWorkModel and profit calculator are provided.")

        # The following arguments are not needed if bayes is False
        if default_bayes:
            add_store('bayes_k0', type=float, default=0.1,
                      help="Variance in prior of the constant \
in the weak convergence model. Not needed if -bayesian is False.")
            add_store('bayes_k1', type=float, default=0.1,
                      help="Variance in prior of the constant \
in the strong convergence model. Not needed if -bayesian is False.")
            add_store('bayes_w_sig', type=float, default=-1,
                      help="Variance in prior of the power \
in the weak convergence model, negative values lead to disabling the fitting. \
Not needed if -bayesian is False.")
            add_store('bayes_s_sig', type=float, default=-1,
                      help="Variance in prior of the power \
in the weak convergence model, negative values lead to disabling the fitting. \
Not needed if -bayesian is False.")
            add_store('bayes_fit_lvls', type=int, default=1000,
                      help="Maximum number of levels used to fit data. \
Not needed if -bayesian is False.")

        # The following arguments are not always needed, and they have
        # a default value
        if additional:
            add_store('max_TOL', type=float,
                      help="The (approximate) tolerance for \
the first iteration. Not needed if TOLs is provided to doRun.")
            add_store('M0', nargs='+', type=int, default=np.array([1]),
                      action=Store_as_array,
                      help="Initial number of samples used to estimate the \
sample variance on levels when not using the Bayesian estimators. \
Not needed if a profit calculator is provided.")
            add_store('maxM', type=int, default=1000, help="Maximum number of \
samples to compute per call to user function")
            add_store('min_lvls', type=int, default=3,
                      help="The initial number of levels to run \
the first iteration. Not needed if a profit calculator is provided.")
            add_store('max_add_itr', type=int, default=2,
                      help="Maximum number of additonal iterations\
to run when the MIMC is expected to but is not converging.\
Not needed if TOLs is provided to doRun.")
            add_store('r1', type=float, default=2,
                      help="A parameters to control to tolerance sequence \
for tolerance larger than TOL. Not needed if TOLs is provided to doRun.")
            add_store('r2', type=float, default=1.1,
                      help="A parameters to control to tolerance sequence \
for tolerance smaller than TOL. Not needed if TOLs is provided to doRun.")
            add_store('h0inv', type=float, nargs='+', action=Store_as_array,
                      default=2,
                      help="Minimum element size get_geometric_hl. \
Not needed if fnHierarchy is provided.")
        return mimcgrp

    def calcTotalWork(self):
        return np.sum(self.Wl_estimate * self.data.M, axis=0)

    def totalErrorEst(self):
        return self.bias + (self.stat_error if not np.isnan(self.stat_error) else 0)

    def __str__(self):
        output = "Time={:.12e}\nEg={}\n\
Bias={:.12e}\nStatErr={:.12e}\
\nTotalErrEst={:.12e}\n".format(self.data.calcTotalTime(),
                                str(self.data.calcEg()),
                                self.bias,
                                self.stat_error,
                                self.totalErrorEst())
        V = self.Vl_estimate
        Vl = self.fn.Norm(self.data.calcDeltaVl())
        E = self.fn.Norm(self.data.calcDeltaEl())
        T = self.data.calcTl()

        output += ("{:<8}{:^20}{:^20}{:^20}{:>8}{:>15}\n".format(
            "Level", "E", "V", "sampleV", "M", "Time"))
        for i in range(0, len(self.data.lvls)):
            #,100 * np.sqrt(V[i]) / np.abs(E[i])
            output += ("{:<8}{:>+20.12e}{:>20.12e}{:>20.12e}{:>8}{:>15.6e}\n".format(
                str(self.data.lvls[i]), E[i], V[i], Vl[i], self.data.M[i], T[i]))
        return output

    def fnNorm1(self, x):
        """ Helper function to return norm of a single element
        """
        return self.fn.Norm(np.array([x]))[0]

    def _estimateBias(self):
        if not self.params.bayesian:
            return self.data.lvls.estimate_bias(self.fn.Norm(self.data.calcDeltaEl()))

            # if np.sum(bnd) == len(self.data.lvls):
            #     return np.inf
            # bnd_val = self.data[bnd].calcDeltaEl()
            # if self.params.abs_bnd:
            #     return np.sum(self.fn.Norm(bnd_val))
            # return self.fnNorm1(np.sum(bnd_val))
        return self._estimateBayesianBias()

    ################## Bayesian specific functions
    def _estimateBayesianBias(self, L=None):
        L = L or len(self.all_data.lvls)-1
        if L <= 1:
            raise Exception("Must have at least 2 levels")
        return self.Q.W * self._get_hl(L)[-1]**self.Q.w[0]

    def _get_hl(self, L):
        lvls = np.arange(0, L+1).reshape((-1, 1))
        return  self.fn.Hierarchy(lvls=lvls).reshape(1, -1)[0]

    def _estimateBayesianVl(self, L=None):
        if np.sum(self.all_data.M, axis=0) == 0:
            return self.fn.Norm(self.all_data.calcDeltaVl())
        oL = len(self.all_data.lvls)-1
        L = L or oL
        if L <= 1:
            raise Exception("Must have at least 2 levels")
        included = np.nonzero(np.logical_and(self.all_data.M > 0,
                                             np.arange(0,
                                                       len(self.all_data.lvls))
                                             >= 1))[0]
        hl = self._get_hl(L)
        M = self.all_data.M[included]
        s1 = self.all_data.psums_delta[included, 0]
        m1 = self.all_data[included].calcDeltaEl()
        s2 = self.all_data.psums_delta[included, 1]
        mu = self.Q.W*(hl[included-1]**self.Q.w[0] - hl[included]**self.Q.w[0])

        Lambda = 1./(self.Q.S*(hl[:-1]**(self.Q.s[0]/2.) -
                               hl[1:]**(self.Q.s[0]/2.))**2)

        tmpM = np.concatenate((self.all_data.M[1:], np.zeros(L-oL)))
        G_3 = self.params.bayes_k1 * Lambda + tmpM/2.0
        G_4 = self.params.bayes_k1*np.ones(L+1)
        G_4[included] += 0.5*(self.fn.Norm(s2 - s1*m1*2 + s1*m1) + \
                              M*self.params.bayes_k0*(
                                  self.fn.Norm(m1)-mu)**2/
                              (self.params.bayes_k0+M) )

        Vl_estimate = np.concatenate((
            self.fn.Norm(self.all_data[0].calcDeltaVl()),G_4[1:] / G_3))
        # Vl_sample = self.all_data.calcDeltaVl()
        # Vl_estimate[:len(Vl_sample)] = Vl_sample
        return Vl_estimate

    def _estimateQParams(self):
        if not self.params.bayesian:
            return
        if np.sum(self.all_data.M, axis=0) == 0:
            return   # Cannot really estimate anything without at least some samples
        L = len(self.all_data.lvls)-1
        if L <= 1:
            raise Exception("Must have at least 2 levels")
        hl = self._get_hl(L)
        included = np.nonzero(\
                np.logical_and(self.all_data.M > 0,
                               np.arange(0, len(self.all_data.lvls))
                               >= np.maximum(1, L-self.params.bayes_fit_lvls)))[0]
        M = self.all_data[included].M
        s1 = self.all_data.psums_delta[included, 0]
        s2 = self.all_data.psums_delta[included, 1]
        t1 = hl[included-1]**self.Q.w[0] - hl[included]**self.Q.w[0]
        t2 = (hl[included-1]**(self.Q.s[0]/2.) - hl[included]**(self.Q.s[0]/2.))**-2

        self.Q.W = self.fnNorm1(np.sum(s1 * _expand(t1, 0, s1.shape) *
                                       _expand(t2, 0, s1.shape), axis=0) /
                                np.sum(M * t1**2 * t2, axis=0) )
        self.Q.S = (np.sum(self.fn.Norm(s2* _expand(t2, 0, s2.shape)) -
                           self.fn.Norm(s1*_expand(self.Q.W*t1*2*t2,
                                                 0, s1.shape)), axis=0) +
                    np.sum(M*self.Q.W**2*t1**2*t2)) / np.sum(M)
        if self.params.bayes_w_sig > 0 or self.params.bayes_s_sig > 0:
            # TODO: Estimate w=q_1, s=q_2
            raise NotImplemented("TODO, estimate w and s")

    def _estimateOptimalL(self, TOL):
        assert self.params.bayesian, "MIMC should be Bayesian to \
estimate optimal number of levels"
        minL = len(self.data)
        minWork = np.inf
        LsRange = range(len(self.data.lvls),
                        len(self.data.lvls)+1+self.params.incL)
        for L in LsRange:
            bias_est = self._estimateBayesianBias(L)
            if bias_est >= TOL and L < LsRange[-1]:
                continue
            lvls = setutil.VarSizeList(np.arange(0, L+1).reshape((-1, 1)), min_dim=1)
            Wl = self.fn.WorkModel(lvls=lvls)
            M = self._calcTheoryM(TOL,
                                  theta=self._calcTheta(TOL, bias_est),
                                  Vl=self._estimateBayesianVl(L), Wl=Wl)
            totalWork = np.sum(Wl*M)
            if totalWork < minWork:
                minL = L
                minWork = totalWork
        return minL

    ################## END: Bayesian specific function
    def _estimateAll(self):
        self._estimateQParams()
        self.Vl_estimate = self.fn.Norm(self.all_data.calcDeltaVl()) \
                           if not self.params.bayesian \
                           else self._estimateBayesianVl()
        self.Wl_estimate = self.fn.WorkModel(lvls=self.data.lvls)
        self.bias = self._estimateBias()
        from scipy.stats import norm
        Ca = norm.ppf(self.params.confidence)
        self.stat_error = np.inf if np.any(self.data.M == 0) \
                          else Ca * \
                               np.sqrt(np.sum(self.Vl_estimate / self.data.M))

    def _extendLevels(self, new_lvls=None):
        prev = len(self.data.lvls)
        if new_lvls is not None:
            self.data.lvls.add_from_list(new_lvls)
        else:
            self.fn.ExtendLvls(lvls=self.data.lvls)
        assert(prev != len(self.data.lvls))

        self.data._levels_added()
        self.all_data._levels_added()
        newTodoM = self.params.M0
        if len(newTodoM) < len(self.data.lvls):
            newTodoM = np.pad(newTodoM,
                              (0,len(self.data.lvls)-len(newTodoM)), 'constant',
                              constant_values=newTodoM[-1])
        return np.concatenate((self.data.M[:prev], newTodoM[prev:len(self.data.lvls)]))

    def SampleLvl(self, mods, inds, M):
        # fnSampleLvl(inds, M) -> Returns a matrix of size (M, len(ind)) and
        # the time estimate
        calcM = 0
        total_time = 0
        p = np.arange(1, self.data.computedMoments()+1)
        psums_delta = _empty_obj()
        psums_fine = _empty_obj()
        while calcM < M:
            curM = np.minimum(M-calcM, self.params.maxM)
            values, time = self.fn.SampleLvl(inds=inds, M=curM)
            total_time += time
            # psums_delta_j = \sum_{i} (\sum_{k} mod_k values_{i,k})**p_j

            delta = np.sum(values * \
                           _expand(mods, 1, values.shape),
                           axis=1)
            A1 = np.tile(delta, (len(p),) + (1,)*len(delta.shape) )
            A2 = np.tile(values[:, 0], (len(p),) + (1,)*len(delta.shape) )
            B = _expand(p, 0, A1.shape)
            psums_delta += np.sum(A1**B , axis=1)
            psums_fine += np.sum(A2**B, axis=1)
            calcM += values.shape[0]
        return calcM, psums_delta, psums_fine, total_time

    def _genSamples(self, totalM, verbose):
        lvls = self.data.lvls
        s = len(lvls)
        t = np.zeros(s)
        active = totalM >= self.data.M
        totalM[active] -= self.data.M[active]
        if np.sum(totalM) == 0:
            return False
        for i in range(0, s):
            if totalM[i] <= 0:
                continue
            if verbose:
                print("# Doing", totalM[i], "of level", lvls[i])
            mods, inds = expand_delta(lvls[i])
            args = self.SampleLvl(mods, inds, totalM[i])

            self.data.addSamples(i, *args)
            if self.all_data != self.data:
                self.all_data.addSamples(i, *args)
        self._estimateAll()
        return True

    def _calcTheta(self, TOL, bias_est):
        if not self.params.const_theta:
            return 1 - bias_est/TOL
        return self.params.theta

    def _calcTheoryM(self, TOL, theta, Vl, Wl, ceil=True, minM=1):
        from scipy.stats import norm
        Ca = norm.ppf(self.params.confidence)
        M = (theta * TOL / Ca)**-2 *\
            np.sum(np.sqrt(Wl * Vl)) * np.sqrt(Vl / Wl)
        M = np.maximum(M, minM)
        M[np.isnan(M)] = minM
        if ceil:
            M = np.ceil(M).astype(np.int)
        return M

    def estimateMonteCarloSampleCount(self, TOL):
        theta = self._calcTheta(TOL, self.bias)
        V = self.Vl_estimate[self.data.lvls.find([])]
        if np.isnan(V):
            return np.nan
        from scipy.stats import norm
        Ca = norm.ppf(self.params.confidence)

        return np.maximum(np.reshape(self.params.M0, (1,))[-1],
                          int(np.ceil((theta * TOL / Ca)**-2 * V)))

    def doRun(self, finalTOL=None, TOLs=None, verbose=None):
        self._checkFunctions()
        finalTOL = finalTOL or self.params.TOL
        if TOLs is None:
            TOLs = [finalTOL] if not hasattr(self.params, "max_TOL") \
                   else get_tol_sequence(finalTOL, self.params.max_TOL,
                                         max_additional_itr=self.params.max_add_itr,
                                         r1=self.params.r1,
                                         r2=self.params.r2)
        if verbose is None:
            verbose = self.params.verbose
        if not all(x >= y for x, y in zip(TOLs, TOLs[1:])):
            raise Exception("Tolerances must be decreasing")

        import time
        tic = time.time()
        self.Q.theta = self.params.theta
        self.bias = np.inf
        self.stat_error = np.inf
        import gc
        def less(a, b, rel_tol=1e-09, abs_tol=0.0):
            return a-b <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        itrIndex = 0
        for TOL in TOLs:
            samples_added = False
            if verbose:
                print("# TOL", TOL)
            while True:
                gc.collect()
                if self.params.bayesian and len(self.data.lvls) > 0:
                    L = self._estimateOptimalL(TOL)
                    if L > len(self.data.lvls):
                        self._extendLevels(new_lvls=np.arange(
                            len(self.data.lvls), L+1).reshape((-1, 1)))
                        self._estimateAll()

                self.Q.theta = np.maximum(self._calcTheta(TOL, self.bias),
                                          self.params.theta)
                if len(self.data.lvls) == 0 or \
                   (not self.params.bayesian and
                    self.bias > (1 - self.params.theta) * TOL):
                    # Bias is not satisfied (or this is the first iteration)
                    # Add more levels
                    newTodoM = self._extendLevels()
                    samples_added = self._genSamples(newTodoM, verbose) or samples_added
                    self.Q.theta = np.maximum(self._calcTheta(TOL, self.bias),
                                              self.params.theta)

                todoM = self._calcTheoryM(TOL, self.Q.theta,
                                          self.Vl_estimate,
                                          self.Wl_estimate)
                if verbose:
                    print("# theta", self.Q.theta)
                    print("# New M: ", todoM)
                if not self.params.reuse_samples:
                    self.data.zero_samples()
                samples_added = self._genSamples(todoM, verbose) or samples_added
                if verbose:
                    print(self, end="")
                    print("------------------------------------------------")
                if self.fn.ItrDone is not None and samples_added:
                    self.fn.ItrDone(iteration_idx=itrIndex,
                                    TOL=TOL,
                                    totalTime=time.time() - tic)
                itrIndex += 1


            if verbose:
                print("{} took {}".format(TOL, time.time()-tic))
                print("################################################")
            if less(TOL, finalTOL) and self.totalErrorEst() <= finalTOL:
                break

@public
def work_estimate(lvls, gamma):
    return np.prod(np.exp(lvls.to_dense_matrix(base=0)*gamma), axis=1)

def expand_delta(lvl):
    """
    This routine takes a multi-index level and produces
    a list of levels and weights that are needed to evaluate
    the multi-dimensional difference estimator.

    For example, in the MLMC setting the function
    x,y = expand_delta([N])

    sets y to an array of [N] and [N-1]
    and x to an array of 1 and -1.

    """
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


@public
def get_geometric_hl(lvls, h0inv, beta):
    return beta**(-np.array(lvls, dtype=np.float))/h0inv


@public
def get_tol_sequence(TOL, maxTOL, max_additional_itr=1, r1=2, r2=1.1):
    # number of iterations until TOL
    eni = int(-(np.log(TOL)-np.log(maxTOL))/np.log(r1))
    return np.concatenate((TOL*r1**np.arange(eni, -1, -1),
                           TOL*r2**-np.arange(1, max_additional_itr+1)))

@public
def get_optimal_hl(mimc):
    # TODO: Get formula from HajiAli 2015, Optimizing MLMC hierarchies
    raise NotImplemented("TODO: get_optimal_hl")


@public
def calcMIMCRate(w, s, gamma):
    d = len(w)
    if len(s) != d or len(gamma) != d:
        raise ValueError("w,s and gamma must have the same size")
    delta = (gamma-s)/(2*w)
    zeta = np.max(delta)
    xi = np.min((2.*w - s) / gamma)
    d2 = np.sum(delta == 0)
    dz = np.sum(delta == zeta)
    rate = -2.*(1. + np.maximum(0, zeta))
    log_rate = np.nan
    if (zeta <= 0 and zeta < xi) or (zeta == xi and zeta == 0 and d <= 2):
        log_rate = 2*d2
    elif zeta > 0 and xi > 0:
        log_rate = 2*(dz-1)*(zeta+1)
    elif zeta == 0 and xi == 0 and d > 2:
        log_rate = 2*d2 + d - 3
    elif zeta > 0 and xi == 0:
        log_rate = d-1 + 2*(dz-1)*(1+zeta)
    return rate, log_rate

def extend_prof_lvls(lvls, profCalc, min_lvls):
    added = 0
    if len(lvls) == 0:
        # add seed
        lvls.add_from_list([[]])
        added += 1
    while added < 1 or (len(lvls) < min_lvls):
        lvls.expand_set(profCalc)
        added += 1
