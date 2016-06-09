from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import itertools
import warnings
from . import setutil

__all__ = []


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


class empty_obj(object):
    def __init__(self, shape):
        self.resize(shape)

    def __add__(self, newarr):
        assert(newarr.shape == self.shape)
        return newarr  # Forgot about this object

    def resize(self, shape, refcheck=False):
        # TODO: Should handle integer weights better than this
        if isinstance(shape, tuple):
            self.shape = shape
        else:
            self.shape = (shape, )


def compute_raw_moments(psums, M):
    '''
    Returns the raw moments or None when M=0.
    '''
    idx = M != 0
    val = np.empty_like(psums)
    val[idx, :] = psums[idx, :] / np.tile(M[idx].reshape((-1,1)),
                                          (1, psums.shape[1]))
    val[M == 0, :] = None
    return val

def compute_central_moment(psums, M, moment):
    '''
    Returns the centralized moments or None when M=0.
    '''
    raw = compute_raw_moments(psums, M)
    if moment == 1:
        return raw[:, 0]
    n = moment
    pn = np.tile(n, raw.shape[0])
    val = (raw[:, 0]**pn) * (-1)**n
    # From http://mathworld.wolfram.com/CentralMoment.html
    nfact = np.math.factorial(n)
    for k in range(1, moment+1):
        nchoosek = nfact / (np.math.factorial(k) * np.math.factorial(n-k))
        val +=  (raw[:, k-1] * raw[:, 0]**(pn-k)) * nchoosek * (-1)**(n-k)
    if moment % 2 == 1:
        return val
    # The moment should be positive
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


@public
class MIMCData(object):

    """
    MIMC Data is a class for describing necessary data
    for a MIMC data, such as the dimension of the problem,
    list of levels, times exerted, sample sizes, etc...

    In a MIMC Run object, the data is stored in a MIMCData object

    """

    def __init__(self, dim, lvls=None, psums_delta=None,
                 psums_fine=None, t=None, M=None, moments=2):
        self.moments = moments
        self.dim = dim
        self.lvls = lvls          # MIMC lvls
        self.psums_delta = psums_delta        # sums of lvls
        self.psums_fine = psums_fine  # sums of lvls
        self.t = t                # Time of lvls
        self.M = M                # Number of samples in each lvl
        if self.lvls is None:
            self.lvls = []
        if self.psums_delta is None:
            self.psums_delta = empty_obj((0, moments))
        if self.psums_fine is None:
            self.psums_fine = empty_obj((0, moments))
        if self.t is None:
            self.t = np.empty(0)
        if self.M is None:
            self.M = np.empty(0, dtype=np.int)

        assert(len(self.lvls) == self.psums_fine.shape[0])
        assert(len(self.lvls) == self.psums_delta.shape[0])
        assert(len(self.lvls) == self.M.shape[0])
        assert(len(self.lvls) == self.t.shape[0])

    def calcEg(self):
        """
        Return the sum of the sample estimators for
        all the levels
        """
        return np.sum(self.calcDeltaEl())

    def __len__(self):
        return len(self.lvls)

    def __getitem__(self, ind):
        return MIMCData(self.dim,
                        lvls=np.array(self.lvls, dtype=object)[ind].reshape((-1,self.dim)).tolist(),
                        psums_delta=self.psums_delta[ind, :].reshape((-1, self.psums_delta.shape[1])),
                        psums_fine=self.psums_fine[ind, :].reshape((-1, self.psums_fine.shape[1])),
                        t=self.t[ind].reshape(-1), M=self.M[ind].reshape(-1))

    def Dim(self):
        return self.dim

    def computedMoments(self):
        return self.moments

    def calcDeltaVl(self):
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
        val = np.zeros_like(self.M, dtype=np.float)
        import IPython
        IPython.embed()
        val[idx] = self.psums_delta[idx, moment-1] / self.M[idx]
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
        return np.sum(self.t)

    def addSamples(self, psums_delta, psums_fine, M, t):
        assert psums_fine.shape[0] == len(M) and psums_delta.shape[0] == len(M) \
            and len(M) == len(t) and np.min(M) >= 0, "Inconsistent arguments "
        self.psums_delta += psums_delta
        self.psums_fine += psums_fine
        self.M += M
        self.t += t

    def zero_samples(self):
        self.M = np.zeros_like(self.M)
        self.t = np.zeros_like(self.t)
        self.psums_delta = empty_obj((len(self.lvls), self.computedMoments()))
        self.psums_fine = empty_obj((len(self.lvls), self.computedMoments()))

    def addLevels(self, new_lvls):
        assert(len(new_lvls) > 0)
        prev = len(self.lvls)
        self.lvls.extend(new_lvls)
        s = len(self.lvls)
        self.psums_delta.resize((s, self.psums_delta.shape[1]), refcheck=False)
        self.psums_fine.resize((s, self.psums_fine.shape[1]), refcheck=False)
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
        self.params = MyDefaultDict(**kwargs)
        self.fnHierarchy = None
        self.fnWorkModel = None
        self.fnSampleLvl = None
        self.fnItrDone = None
        self.fnExtendLvls = None
        self.Vl_estimate = None
        self.Wl_estimate = None
        self.bias = np.inf           # Approximation of the discretization error
        self.stat_error = np.inf     # Sampling error (based on M)
        if old_data is not None:
            assert(old_data.dim == self.params.dim)
            self.all_data = self.data = old_data
        else:
            self.all_data = self.data = MIMCData(dim=self.params.dim, moments=self.params.moments)
            if not self.params.reuse_samples:
                self.all_data = MIMCData(dim=self.params.dim, moments=self.params.moments)

        if (hasattr(self.params, "w") and len(self.params.w) != self.data.dim) or \
           (hasattr(self.params, "s") and len(self.params.s) != self.data.dim) or \
           (hasattr(self.params, "gamma") and len(self.params.gamma) != self.data.dim) or \
           (hasattr(self.params, "beta") and len(self.params.beta) != self.data.dim):
                raise ValueError("Size of beta, w, s and gamma must be of size dim")

        if self.params.bayesian and self.data.dim > 1:
            raise NotImplementedError("Bayesian parameter fitting is only \
supported in one dimensional problem")

        if self.params.bayesian:
            self.Q = MyDefaultDict(S=np.inf, W=np.inf,
                                   w=self.params.w, s=self.params.s,
                                   theta=np.nan)
        else:
            self.Q = MyDefaultDict(theta=np.nan)

    def _checkFunctions(self):
        # If self.params.reuse_samples is True then
        # all_data will always equal data
        if self.fnWorkModel is None and hasattr(self.params, "gamma"):
            self.fnWorkModel = lambda lvls: work_estimate(lvls,
                                                          np.log(self.params.beta) *
                                                          np.array(self.params.gamma))

        if self.fnHierarchy is None:
            self.fnHierarchy = lambda lvls: get_geometric_hl(lvls,
                                                             self.params.h0inv,
                                                             np.array(self.params.beta))

        if self.params.bayesian and self.fnWorkModel is None:
            raise NotImplementedError("Bayesian parameter fitting is only \
supported with a given work model")

        if self.fnWorkModel is None:
            # ADDING WORK MODEL B
            warnings.warn("fnWorkModel is not provided, using run-time estimates.")
            raise NotImplemented("Need to check that the lvls \
are the same as the argument ones")
            self.fnWorkModel = lambda lvls: self.Tl()
        # self.fnExtendLvls = self.fnExtendLvls or \
        #                     (lambda: extend_lvls_tensor(self.data.dim,
        #                                                 self.data.lvls,
        #                                                 self.params.M0,
        #                                                 self.params.min_lvls/self.params.dim))
        if self.fnExtendLvls is None:
            weights = np.array(self.params.beta) * (np.array(self.params.w) +
                                                    (np.array(self.params.s) -
                                                     np.array(self.params.gamma))/2.)
            weights /= np.sum(weights)
            if len(weights) == 1:
                weights = weights[0]*np.ones(self.params.dim)

            self.fnExtendLvls = lambda w=weights: extend_lvls_td(w,
                                                                 self.data.lvls,
                                                                 self.params.M0,
                                                                 self.params.min_lvls/self.params.dim)

        if self.fnSampleLvl is None:
            raise ValueError("Must set the sampling functions fnSampleLvl")

    def setFunctions(self, **kwargs):
        # fnExtendLvls(): Returns new lvls and number of samples on each.
        #    called only once if the Bayesian method is used
        # fnSampleLvl(moments, mods, inds, M):
        #    Returns M, array: M sums of mods*inds, and total
        #    (linear) time it took to compute them
        # fnItrDone(i, TOLs, totalTime): Called at the end of iteration
        #    i out of TOLs
        # fnWorkModel(lvls): Returns work estimate of lvls
        # fnHierarchy(lvls): Returns associated hierarchy of lvls
        for k in kwargs.keys():
            if k not in ["fnExtendLvls", "fnSampleLvl",
                         "fnItrDone", "fnWorkModel",
                         "fnHierarchy", "fnSampleQoI"]:
                raise KeyError("Invalid function name")
            setattr(self, k, kwargs[k])

    @staticmethod
    def addOptionsToParser(parser, pre='-mimc_', additional=True, default_bayes=True):
        def str2bool(v):
            # susendberg's function
            return v.lower() in ("yes", "true", "t", "1")
        mimcgrp = parser.add_argument_group('MIMC', 'Arguments to control MIMC logic')
        mimcgrp.register('type', 'bool', str2bool)

        def add_store(name, **kwargs):
            if "default" in kwargs and "help" in kwargs:
                kwargs["help"] += " (default: {})".format(kwargs["default"])
            mimcgrp.add_argument(pre + name, dest=name,
                                 action="store",
                                 **kwargs)

        add_store('verbose', type='bool', default=False,
                  help="Verbose output")
        add_store('bayesian', type='bool', default=False,
                  help="Use Bayesian fitting to estimate bias, variance and optimize number \
of levels in every iteration. This is based on CMLMC.")
        add_store('dim', type=int, help="Number of dimensions used in MIMC")
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
        add_store('w', nargs='+', type=float,
                  help="Weak convergence rates. Must be scalar or of size -dim. \
Not needed if fnExtendLvls is specified and -bayesian is False.")
        add_store('s', nargs='+', type=float,
                  help="Strong convergence rates. Must be a scalar or of size -dim. \
Not needed if fnExtendLvls is specified and -bayesian is False.")
        add_store('TOL', type=float,
                  help="The required tolerance for the MIMC run")
        add_store('beta', type=float, nargs='+',
                  help="Level separation parameter. to be used \
with get_geometric_hl. Not needed if fnHierarchy is provided.")
        add_store('gamma', type=float, nargs='+',
                  help="Work exponent to be used with work_estimate.\
Not needed if fnWorkModel and fnExtendLvls are provided.")

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
            add_store('max_TOL', type=float, default=0.1,
                      help="The (approximate) tolerance for \
the first iteration. Not needed if TOLs is provided to doRun.")
            add_store('M0', nargs='+', type=int, default=[10],
                      help="Initial number of samples used to estimate the \
sample variance on levels when not using the Bayesian estimators. \
Not needed if fnExtendLvls is provided.")
            add_store('maxM', type=int, default=1000, help="Maximum number of \
samples to compute per call to user function")
            add_store('min_lvls', type=int, default=2,
                      help="The initial number of levels to run \
the first iteration. Not needed if fnExtendLvls is provided.")
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
            add_store('h0inv', type=float, nargs='+', default=2,
                      help="Minimum element size get_geometric_hl. \
Not needed if fnHierarchy is provided.")
        return mimcgrp

    def calcTotalWork(self):
        return np.sum(self.Wl_estimate * self.data.M)

    def totalErrorEst(self):
        return self.bias + self.stat_error

    def __str__(self):
        output = "Time={:.12e}\nEg={:.12e}\n\
Bias={:.12e}\nStatErr={:.12e}\
\nTotalErrEst={:.12e}\n".format(self.data.calcTotalTime(),
                                self.data.calcEg(),
                                self.bias,
                                self.stat_error,
                                self.totalErrorEst())
        V = self.Vl_estimate
        Vl = self.data.calcDeltaVl()
        E = self.data.calcDeltaEl()
        T = self.data.calcTl()

        output += ("{:<8}{:^20}{:^20}{:^20}{:>8}{:>15}\n".format(
            "Level", "E", "V", "sampleV", "M", "Time"))
        for i in range(0, len(self.data.lvls)):
            assert(V[i]>=0)
            #,100 * np.sqrt(V[i]) / np.abs(E[i])
            output += ("{:<8}{:>+20.12e}{:>20.12e}{:>20.12e}{:>8}{:>15.6e}\n".format(
                str(self.data.lvls[i]), E[i], V[i], Vl[i], self.data.M[i], T[i]))
        return output

    ################## Bayesian specific functions
    def _estimateBias(self):
        if not self.params.bayesian:
            bnd = is_boundary(self.data.dim, self.data.lvls)
            if np.sum(bnd) == len(self.data.lvls):
                return np.inf
            bnd_val = self.data[bnd].calcDeltaEl()
            if self.params.abs_bnd:
                return np.abs(np.sum(np.abs(bnd_val)))
            return np.abs(np.sum(bnd_val))
        return self._estimateBayesianBias()

    def _estimateBayesianBias(self, L=None):
        L = L or len(self.all_data.lvls)-1
        if L <= 1:
            raise Exception("Must have at least 2 levels")
        hl = self.fnHierarchy(lvls=np.arange(0, L+1).reshape((-1, 1))).reshape(1, -1)[0]
        return self.Q.W * hl[-1]**self.Q.w[0]

    def _estimateBayesianVl(self, L=None):
        if np.sum(self.all_data.M) == 0:
            return self.all_data.calcDeltaVl()
        oL = len(self.all_data.lvls)-1
        L = L or oL
        if L <= 1:
            raise Exception("Must have at least 2 levels")
        hl = self.fnHierarchy(lvls=np.arange(0, L+1).reshape((-1, 1))).reshape(1, -1)[0]
        M = np.concatenate((self.all_data[1:].M, np.zeros(L-oL)))
        s1 = np.concatenate((self.all_data.psums_delta[1:, 0], np.zeros(L-oL)))
        m1 = np.concatenate((self.all_data[1:].calcDeltaEl(), np.zeros(L-oL)))
        s2 = np.concatenate((self.all_data.psums_delta[1:, 1],
                             np.zeros(L-oL)))
        mu = self.Q.W*(hl[:-1]**self.Q.w[0] - hl[1:]**self.Q.w[0])
        Lambda = 1./(self.Q.S*(hl[:-1]**(self.Q.s[0]/2.) - hl[1:]**(self.Q.s[0]/2.))**2)
        G_3 = self.params.bayes_k1 * Lambda + M/2.0
        # G_4 = self.params.bayes_k1 + \
        #       0.5*M*(s2 + self.params.bayes_k0 * (m1 - mu)**2 /
        #              (self.params.bayes_k0 + M)) - 0.5*m1**2
        G_4 = self.params.bayes_k1 + \
              0.5*(s2 -2*s1*m1 + s1*m1 +
                   M*self.params.bayes_k0*(m1-mu)**2 / (self.params.bayes_k0+M) )
        Vl_estimate = np.concatenate((self.all_data[0].calcDeltaVl(), G_4 / G_3))
        # Vl_sample = self.all_data.calcDeltaVl()
        # Vl_estimate[:len(Vl_sample)] = Vl_sample
        return Vl_estimate

    def _estimateQParams(self):
        if not self.params.bayesian:
            return
        if np.sum(self.all_data.M) == 0:
            return   # Cannot really estimate anything without at least some samples
        L = len(self.all_data.lvls)-1
        if L <= 1:
            raise Exception("Must have at least 2 levels")
        hl = self.fnHierarchy(lvls=np.arange(0, L+1).reshape((-1, 1))).reshape(1, -1)[0]
        begin = np.maximum(1, L-self.params.bayes_fit_lvls)
        M = self.all_data[begin:].M
        # m1 = self.all_data[begin:].calcDeltaEl()
        # m2 = self.all_data[begin:].calcDeltaEl(moment=2)
        # wl = hl[begin:]**self.Q.w[0] - hl[(begin-1):-1]**self.Q.w[0]
        # sl = (hl[begin:]**(self.Q.s[0]/2.) - hl[(begin-1):-1]**(self.Q.s[0]/2.))**-2
        # self.Q.W = np.abs(np.sum(wl * sl * M * m1) / np.sum(M * wl**2 * sl))
        # self.Q.S = np.sum(sl * (m2 - 2*m1*self.Q.W*wl + self.Q.W**2*wl**2)) / np.sum(M)
        s1 = self.all_data.psums_delta[begin:, 0]
        s2 = self.all_data.psums_delta[begin:, 1]
        t1 = hl[(begin-1):-1]**self.Q.w[0] - hl[begin:]**self.Q.w[0]
        t2 = (hl[(begin-1):-1]**(self.Q.s[0]/2.) - hl[begin:]**(self.Q.s[0]/2.))**-2
        self.Q.W = np.abs(np.sum(s1 * t1 * t2) / np.sum(M * t1**2 * t2))
        self.Q.S = np.sum(t2*(s2 - 2*s1*t1*self.Q.W + M*self.Q.W**2*t1**2)) / np.sum(M)
        if self.params.bayes_w_sig > 0 or self.params.bayes_s_sig > 0:
            # TODO: Estimate w=q_1, s=q_2
            raise NotImplemented("TODO, estimate w and s")

    def _estimateOptimalL(self, TOL):
        assert self.params.bayesian, "MIMC should be Bayesian to \
estimate optimal number of levels"
        minL = len(self.data)
        minWork = np.inf
        LsRange = range(len(self.data.lvls), len(self.data.lvls)+1+self.params.incL)
        for L in LsRange:
            bias_est = self._estimateBayesianBias(L)
            if bias_est >= TOL and L < LsRange[-1]:
                continue
            Wl = self.fnWorkModel(lvls=np.arange(0, L+1).reshape((-1, 1)))
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
        self.Vl_estimate = self.all_data.calcDeltaVl() if not self.params.bayesian \
                           else self._estimateBayesianVl()
        self.Wl_estimate = self.fnWorkModel(lvls=self.data.lvls)
        self.bias = self._estimateBias()
        from scipy.stats import norm
        Ca = norm.ppf(self.params.confidence)
        self.stat_error = np.inf if np.any(self.data.M == 0) \
                          else Ca * \
                               np.sqrt(np.sum(self.Vl_estimate / self.data.M))

    def _addLevels(self, lvls):
        self.data.addLevels(lvls)
        if self.all_data != self.data:
            self.all_data.addLevels(lvls)


    def SampleLvl(self, mods, inds, M):
        # fnSampleLvl(inds, M) -> Returns a matrix of size (M, len(ind)) and
        # the time estimate
        calcM = 0
        total_time = 0
        p = np.arange(1, self.data.computedMoments()+1)
        psums_delta = empty_obj(len(p))
        psums_fine = empty_obj(len(p))
        while calcM < M:
            curM = np.minimum(M-calcM, self.params.maxM)
            values, time = self.fnSampleLvl(inds=inds, M=curM)
            total_time += time
            # TODO: Consider moving to C/C++
            # psums_delta_j = \sum_{i} (\sum_{k} mod_k values_{i,k})**p_j

            A1 = np.tile(np.sum(values*np.tile(mods, (values.shape[0], 1)), axis=1), (len(p), 1))
            A2 = np.tile(values[:, 0], (len(p), 1))
            B = np.tile(p, (len(values),1)).transpose()
            psums_delta += np.sum(A1**B , axis=1)
            psums_fine += np.sum(A2**B, axis=1)
            calcM += values.shape[0]
        return calcM, psums_delta, psums_fine, total_time

    def _genSamples(self, totalM, verbose):
        lvls = self.data.lvls
        s = len(lvls)
        M = np.zeros(s, dtype=np.int)
        # TODO: Better way?
        psums_delta = np.array([None]*np.prod((s, self.data.computedMoments()))).reshape((s, self.data.computedMoments()))
        psums_fine = np.array([None]*np.prod((s, self.data.computedMoments()))).reshape((s, self.data.computedMoments()))
        t = np.zeros(s)
        for i in range(0, s):
            if totalM[i] <= self.data.M[i]:
                continue
            if verbose:
                print("# Doing", totalM[i]-self.data.M[i], "of level", lvls[i])
            mods, inds = lvl_to_inds_general(lvls[i])
            M[i], psums_delta[i, :], psums_fine[i, :],  t[i] = self.SampleLvl(mods, inds,
                                                                      totalM[i] - self.data.M[i])
        self.data.addSamples(psums_delta, psums_fine, M, t)
        if self.all_data != self.data:
            self.all_data.addSamples(psums_delta, psums_fine, M, t)
        self._estimateAll()

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
        if ceil:
            M = np.ceil(M).astype(np.int)
        return M

    def estimateMonteCarloSampleCount(self, TOL):
        theta = np.maximum(self._calcTheta(TOL, self.bias), self.params.theta)
        from scipy.stats import norm
        Ca = norm.ppf(self.params.confidence)
        return np.maximum(np.reshape(self.params.M0, (1,))[-1],
                          int(np.ceil((theta * TOL / Ca)**-2 * self.Vl_estimate[0])))

    def doRun(self, finalTOL=None, TOLs=None, verbose=None):
        self._checkFunctions()
        finalTOL = finalTOL or self.params.TOL
        TOLs = TOLs or get_tol_sequence(finalTOL, self.params.max_TOL,
                                        max_additional_itr=self.params.max_add_itr,
                                        r1=self.params.r1,
                                        r2=self.params.r2)
        if verbose is None:
            verbose = self.params.verbose
        if len(self.data.lvls) != 0:
            warnings.warn("Running the same object twice, resetting")
            self.data = MIMCData(self.data.dim)
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

        for itrIndex, TOL in enumerate(TOLs):
            if verbose:
                print("# TOL", TOL)
            while True:
                gc.collect()
                if self.params.bayesian and len(self.data.lvls) > 0:
                    L = self._estimateOptimalL(TOL)
                    if L > len(self.data.lvls):
                        self._addLevels(np.arange(len(self.data.lvls),
                                                  L+1).reshape((-1, 1)))
                        self._estimateAll()

                self.Q.theta = np.maximum(self._calcTheta(TOL, self.bias),
                                          self.params.theta)
                if len(self.data.lvls) == 0 or \
                   (not self.params.bayesian and self.bias > (1 - self.Q.theta) * TOL):
                    # Bias is not satisfied (or this is the first iteration)
                    # Add more levels
                    newlvls, newTodoM = self.fnExtendLvls()
                    prev = len(self.data.lvls)
                    self._addLevels(newlvls)
                    self._genSamples(np.concatenate((self.data.M[:prev],
                                                     newTodoM)), verbose)
                    self._estimateAll()
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
                self._genSamples(todoM, verbose)
                if verbose:
                    print(self, end="")
                    print("------------------------------------------------")
                if self.params.bayesian or self.totalErrorEst() < TOL:
                    break

            totalTime = time.time() - tic
            tic = time.time()
            if verbose:
                print("{} took {}".format(TOL, totalTime))
                print("################################################")
            if self.fnItrDone:
                self.fnItrDone(iteration_idx=itrIndex, TOL=TOL, totalTime=totalTime)
            if less(TOL, finalTOL) and self.totalErrorEst() <= finalTOL:
                break

@public
def extend_lvls_tensor(dim, lvls, M0, min_deg=1):
    if len(lvls) <= 0:
        out_lvls = [[0] * dim]
        seeds = lvls = out_lvls
        deg = 0
    else:
        out_lvls = list()
        deg = np.max([np.max(ll) for ll in lvls])
        seeds = [ll for ll in lvls if np.max(ll) == deg]

    additions = [f for f in itertools.product([0, 1], repeat=dim) if max(f) > 0]
    while True:
        newlvls = list()
        for l in seeds:
            newlvls.extend([(np.array(l) + a).tolist() for a in
                            additions if (np.array(l) + a).tolist()
                            not in newlvls])
        out_lvls.extend(newlvls)
        deg += 1
        if deg >= min_deg:
            break
        seeds = newlvls
    if len(M0) < len(lvls): # Exhausted
        return out_lvls, M0[-1]*np.ones(len(out_lvls), dtype=np.int)
    M = np.pad(M0, (0,len(lvls) + len(out_lvls)), 'constant', constant_values=M[-1])
    return out_lvls, M[len(lvls):]

@public
def extend_lvls_td(w, lvls, M0, min_deg=2):
    # w specifies the dimension
    prev_deg = np.max(np.sum(np.array(
        [w*np.array(l) for l in lvls]), axis=1)) if lvls else 0
    max_deg = prev_deg
    while True:
        max_deg += np.min(w)
        max_deg = np.maximum(max_deg, min_deg)
        C, _ = setutil.AnisoProfCalculator(w*0, w).GetIndexSet(max_deg)
        all_lvls = C.to_dense_matrix() - 1
        newlvls = [lvl.tolist() for lvl in all_lvls if lvl.tolist()
                   not in lvls]
        if len(newlvls) > 0:
            if len(M0) < len(lvls): # Exhausted
                return newlvls, M0[-1]*np.ones(len(newlvls), dtype=np.int)
            M = np.pad(M0, (0,len(lvls) + len(newlvls)), 'constant', constant_values=M0[-1])
            return newlvls, M[len(lvls):]


@public
def work_estimate(lvls, gamma):
    return np.prod(np.exp(np.array(lvls)*gamma), axis=1)


def is_boundary(d, lvls):
    if len(lvls) == 1:
        return [True]   # Special case for zero element
    bnd = np.zeros(len(lvls), dtype=int)
    for i in range(0, d):
        x = np.zeros(d)
        x[i] = 1
        bnd += np.array([1 if l[i] == 0 or (np.array(l)+x).tolist() in lvls else 0 for l in lvls])
    return bnd < d


def lvl_to_inds_general(lvl):

    """
    This routine takes a multi-index level and produces
    a list of levels and weights that are needed to evaluate
    the multi-dimensional difference estimator.

    For example, in the MLMC setting the function
    x,y = lvl_to_inds_general([N])

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
    if mimc.data.dim != 1:
        raise NotImplemented("Optimized hierarchies are only supported\
 for one-dimensional problems")

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
