from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ctypes as ct
import numpy.ctypeslib as npct

__arr_double_1__ = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
__arr_uint_1__ = npct.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')

__lib__ = npct.load_library("_kuramoto.so", ".")
__lib__.MultiKuramoto_QoI.restype = None
__lib__.MultiKuramoto_QoI.argtypes = [ct.c_uint32, ct.c_bool, ct.c_bool,
                                      __arr_uint_1__, __arr_uint_1__,
                                      ct.c_uint32, ct.c_double, ct.c_double,
                                      ct.c_double, __arr_double_1__,
                                      __arr_double_1__, __arr_double_1__,
                                      __arr_double_1__]

__lib__.CreateRandGen.restype = ct.c_voidp
__lib__.CreateRandGen.argtypes = [ct.c_uint64]

__lib__.FreeRandGen.restype = None
__lib__.FreeRandGen.argtypes = [ct.c_voidp]

__lib__.SampleKuramoto_QoI.restype = None
__lib__.SampleKuramoto_QoI.argtypes = [ct.c_voidp, ct.c_uint32, ct.c_bool,
                                       ct.c_bool, __arr_uint_1__,
                                       __arr_uint_1__, ct.c_uint32,
                                       ct.c_double, ct.c_double,
                                       ct.c_double, ct.c_uint32,
                                       __arr_double_1__]

__lib__.SampleKuramoto_Cov.restype = None
__lib__.SampleKuramoto_Cov.argtypes = [ct.c_voidp, ct.c_uint32, ct.c_bool,
                                       ct.c_bool, __arr_uint_1__,
                                       __arr_uint_1__, ct.c_uint32,
                                       ct.c_double, ct.c_double,
                                       ct.c_double, ct.c_uint32,
                                       __arr_double_1__]

__lib__.SolveFokkerPlanck1D.restype = None
__lib__.SolveFokkerPlanck1D.argtypes = [ct.c_double, ct.c_double,
                                        ct.c_double, ct.c_uint32,
                                        ct.c_double, ct.c_uint32,
                                        __arr_double_1__]


def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]

    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"
    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
                  return_index=return_index,
                  return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

def Solve(xmax, sig, K, Nx, Nt, T=1):
    x = np.linspace(-xmax, xmax, Nx)
    dx = x[1]-x[0]
    pn = (1./(0.2*np.sqrt(2*np.pi))) * np.exp(-x**2/(2*0.04))

    __lib__.SolveFokkerPlanck1D(sig, K, xmax, Nx, T, Nt, pn)

    return np.sum(np.cos(x) * pn * dx)

def SolveFor(i):
    print("Doing", i)
    t = time.time()
    val = Solve(sig=0.4, K=0.4, T=1, xmax=5+i, Nx=2**i, Nt=2**i)
    data.append(val)
    print("Done", i, "->", val, "in", (time.time()-t)/60.)
    return val

if __name__ == "__main__":
    import time
    data = [1.0620604644297862, 1.012705551389204, 0.9765557373630317,
            0.95719569711857733, 0.94697537016160072, 0.9415663432121697,
            0.93869793834456428, 0.937176792141, 0.93637119039]
    from multiprocessing import Pool
    pool = Pool(10)
    vals = pool.map(SolveFor, range(6+len(data), 20))
    data.append(vals)
    print(data)

def Kuramoto(Ps, Ns, mods, disorder=None, wiener=None, initial=None, T=1, K=1,
             sig=1, var_sig=False, antithetic=True, dim=1):
    assert Ps.shape[0] == Ns.shape[0] and Ps.shape[0] == mods.shape[0], \
        "Ps, Ns and mods must have the same length"
    if wiener is None:
        wiener = np.sqrt(T/np.max(Ns)) * \
                 np.random.standard_normal(dim*np.max(Ps)*np.max(Ns))
    else:
        assert wiener.shape[0] == dim*np.max(Ps)*np.max(Ns), \
            "Must have dim*max(P)*max(N) Weiner increments"

    if disorder is None:
        disorder = np.random.uniform(-1, 1, size=dim*np.max(Ps))
    else:
        assert disorder.shape[0] == dim*np.max(Ps), \
            "Must have the dim*max(P) disorder numbers"

    if initial is None:
        initial = np.random.uniform(-np.pi, np.pi, dim*np.max(Ps))
    else:
        assert initial.shape[0] == dim*np.max(Ps), \
            "Must have the dim*max(P) disorder numbers"

    out = np.empty(Ps.shape[0])
    __lib__.MultiKuramoto_QoI(dim, var_sig, antithetic, Ps, Ns,
                              Ps.shape[0], T, K, sig, initial, disorder,
                              wiener, out)
    return np.sum(mods*out), out


def SampleKuramoto(gen, Ps, Ns, M, T=1, K=1, sig=1,
                   var_sig=False, antithetic=True, dim=1):
    assert Ps.shape[0] == Ns.shape[0], \
        "Ps, Ns and mods must have the same length"
    samples = np.empty(Ps.shape[0] * M)
    __lib__.SampleKuramoto_QoI(gen._handle, dim, var_sig, antithetic,
                               Ps.astype(np.uint32),
                               Ns.astype(np.uint32),
                               Ps.shape[0], T, K, sig, M, samples)
    #return samples.reshape((Ps.shape[0], M)).transpose()
    return samples.reshape((M, Ps.shape[0]))

def SampleKuramoto_Cov(gen, Ps, Ns, M, T=1, K=1, sig=1, var_sig=False,
                   antithetic=True, dim=1):
    assert Ps.shape[0] == Ns.shape[0], \
        "Ps, Ns must have the same length"
    moments = np.empty(2)
    __lib__.SampleKuramoto_Cov(gen._handle, dim, var_sig, antithetic, Ps,
                               Ns, Ps.shape[0], T, K, sig, M,
                               moments)
    return moments


class RandGen(object):
    def __init__(self, seed):
        self._handle = __lib__.CreateRandGen(seed)

    def __del__(self):
        __lib__.FreeRandGen(self._handle)


class SField_Kuramoto(object):
    def __init__(self, K, T, sig,
                 #P_beta=2, N_beta=2,
                 #P0=2, N0=2,
                 beta=np.array([2, 2]), h0=np.array([2, 2]),
                 var_sig=False, antithetic=True):
        assert(len(beta) == len(h0) and (len(beta) == 2 or len(beta) == 3))
        self.K = K
        self.T = T
        self.sig = sig
        self.beta = beta
        self.h0 = h0
        self.P = None
        self.N = None
        self.mods = None
        self.var_sig = var_sig
        self.antithetic = antithetic
        self.dim = len(beta)

    def BeginRuns(self, mods, ind):
        ind = np.array(ind)
        if self.dim == 2:
            self.P = self.h0[0]*self.beta[0]**np.array(ind[:, 0], dtype=np.uint32)
            self.N = self.h0[1]*self.beta[1]**np.array(ind[:, 1], dtype=np.uint32)
        else:
            self.P = self.h0[0]*self.beta[0]**np.array(ind[:, 0], dtype=np.uint32)
            self.P *= self.h0[1]*self.beta[1]**np.array(ind[:, 1], dtype=np.uint32)
            self.N = self.h0[2]*self.beta[2]**np.array(ind[:, 2], dtype=np.uint32)
            self.mods = np.array(mods, dtype=np.float)
            PN, I = unique_rows(np.vstack((self.P, self.N)).transpose(),
                                return_inverse=True)
            self.P = PN[:, 0].copy()
            self.N = PN[:, 1].copy()
            self.mods = np.bincount(I, weights=self.mods)

    def Sample(self, gen):
        initial = np.random.normal(0, scale=0.2, size=np.max(self.P))
        disorder = gen.uniform(-.2, .2, size=np.max(self.P))
        # initial = np.zeros(np.max(self.P))
        wiener = np.sqrt(self.T/np.max(self.N)) * \
                 gen.standard_normal(np.max(self.P)*np.max(self.N))
        return Kuramoto(Ps=self.P, Ns=self.N, mods=self.mods,
                        initial=initial, wiener=wiener,
                        disorder=disorder, K=self.K, T=self.T,
                        sig=self.sig, var_sig=self.var_sig,
                        antithetic=self.antithetic)[0]

    def SampleMultiple(self, gen, M):
        moments = SampleKuramoto(gen, Ps=self.P, Ns=self.N,
                                 mods=self.mods, M=M, K=self.K, T=self.T,
                                 sig=self.sig, var_sig=self.var_sig,
                                 antithetic=self.antithetic)[-2:]
        return moments

    def EndRuns(self):
        return

    def __exit__(self, type, value, traceback):
        return

    def __enter__(self):
        return self

    @staticmethod
    def Init():
        return

    @staticmethod
    def Final():
        return

    def GetDim(self):
        return self.dim


def SampleParallelKuramoto(args):
    import time
    t = time.time()
    seed, dim, Ps, Ns, mods, T, M, K, sig, antithetic, var_sig = args
    gen = RandGen(seed)
    moments = SampleKuramoto(gen, dim=dim, Ps=Ps, Ns=Ns, mods=mods,
                             M=M, T=T, K=K, sig=sig, antithetic=antithetic,
                             var_sig=var_sig)
    return np.hstack((moments, [(time.time()-t)]))

def MC(seed, P, N, M, T, K, sig, var_sig):
    gen = RandGen(seed)
    moments = SampleKuramoto(gen, dim=1, Ps=np.array([P], dtype=np.uint32),
                             Ns=np.array([N], dtype=np.uint32),
                             mods=np.array([1.]), M=M, T=T, K=K,
                             sig=sig, antithetic=False,
                             var_sig=var_sig)
    return moments

def MC_covariance(seed, Ps, N, M, T, K, sig, var_sig):
    gen = RandGen(seed)
    moments = SampleKuramoto_Cov(gen, dim=1, Ps=np.array(Ps, dtype=np.uint32),
                                 Ns=np.array([N]*len(Ps), dtype=np.uint32),
                                 M=M, T=T, K=K, sig=sig, antithetic=True,
                                 var_sig=var_sig)
    return moments
