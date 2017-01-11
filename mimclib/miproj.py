from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from . import setutil
from scipy.linalg import solve
import itertools
import warnings
import time
from . import mimc

__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym

import os
import ctypes as ct
import numpy.ctypeslib as npct
__lib__ = npct.load_library("libset_util", __file__)
__lib__.sample_optimal_leg_pts.restype = None
__lib__.sample_optimal_leg_pts.argtypes = [ct.c_uint32, ct.c_voidp,
                                           npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS'),
                                           ct.c_double, ct.c_double]

@public
def sample_optimal_leg_pts(N, bases_indices, interval=(-1, 1)):
    max_dim = bases_indices.max_dim()
    X = np.empty(max_dim*N)
    __lib__.sample_optimal_leg_pts(N, bases_indices._handle, X,
                                   interval[0], interval[1])
    X = X.reshape((N, max_dim))
    if X.shape[0] == 0:
        W = np.zeros(0)
    else:
        B = TensorExpansion.evaluate_basis(lambda x, n, ii=interval:
                                           legendre_polynomials(x, n, interval=ii),
                                           bases_indices, X)
        W = len(bases_indices) / np.sum(np.power(B, 2), axis=1)
    return X, W

"""
TensorExpansion is a simple object representing a basis function and a list of
coefficients. It assumes that the basis is orthonormal
"""
@public
class TensorExpansion(object):
    def __init__(self, fnBasis, base_indices, coefficients):
        # fnBasis takes (X, n) where X is a list of 1-D points and n is the
        #         max-degree. Returns list of basis evaluated up to n
        self.fnBasis = fnBasis
        self.base_indices = base_indices
        self.coefficients = coefficients

    def __call__(self, X):
        '''
        Return approximation at specified locations.

        :param X: Locations of evaluations
        :return: Values of approximation at specified locations
        '''
        X = np.array(X)
        if len(X.shape) == 0:
            X = X[None, None] # Scalar
        elif len(X.shape) == 1:
            X = X[:, None] # vector
        return TensorExpansion.evaluate_basis(self.fnBasis,
                                              self.base_indices, X).dot(self.coefficients)

    @staticmethod
    def evaluate_basis(fnBasis, base_indices, X):
        '''
        Evaluates basis polynomials at given sample locations.
        Consistency condition is: fnBasis(X, 0) = 1 for any X

        :param X: Sample locations (M, dim)
        :param base_indices: indices of basis to return (N, up_to_dim)
        :return: Basis polynomials evaluated at X, (M, N)
        :rtype: `len(X) x len(mis)` np.array
        '''
        dim = X.shape[1]
        max_deg = np.max(base_indices.to_dense_matrix(), axis=0)
        rdim = np.minimum(dim, len(max_deg))
        values = np.ones((len(X), len(base_indices)))
        basis_values = np.empty(rdim, dtype=object)
        for d in xrange(0, rdim):
            basis_values[d] = fnBasis(X[:, d], max_deg[d]+1)

        for i, mi in enumerate(base_indices):
            for d, j in enumerate(mi):
                values[..., i] *= basis_values[d][:, j]
        return values

    def norm(self):
        '''
        Return L^2 norm of expansion. Assumes basis is orthonormal
        '''
        return np.sqrt(np.sum(self.coefficients**2))

    def __add__(self, other):
        if not isinstance(other, TensorExpansion):
            raise NotImplementedError();
        result = TensorExpansion(self.fnBasis,
                                 self.base_indices.copy(),
                                 self.coefficients.copy())
        for i, new_ind in enumerate(other.base_indices):
            j = result.base_indices.find(new_ind)
            if j is not None:
                # Not really new
                result.coefficients[j] += other.coefficients[i]
            else:
                # new index
                result.base_indices.add_from_list([new_ind])
                result.coefficients = np.hstack((result.coefficients, other.coefficients[i]))
        return result

    def __mul__(self, scale):
        return TensorExpansion(self.fnBasis,
                               self.base_indices.copy(),
                               self.coefficients*scale)

    def __str__(self):
        return "<Polynomial expansion>"

"""
Maintains polynomial approximation of given function on :math:`[a,b]^d`.
Supposed to take function and maintain polynomial coefficients
"""
@public
class MIWProjSampler(object):
    def __init__(self, d=0,  # d is the spatial dimension
                 fnBasis=None,
                 fnSamplesCount=None,
                 fnSamplePoints=None,
                 fnBasisFromLvl=None,
                 fnWorkModel=None,
                 reuse_samples=False):
        self.fnBasis = fnBasis
        # Returns samples count of a projection index to ensure stability
        self.fnSamplesCount = fnSamplesCount if fnSamplesCount is not None else default_samples_count
        # Returns point sample and their weights
        self.fnSamplePoints = fnSamplePoints
        self.fnBasisFromLvl = fnBasisFromLvl if fnBasisFromLvl is not None else default_basis_from_level
        self.d = d   # Spatial dimension
        self.alpha_ind = np.zeros(0)
        self.fnWorkModel = fnWorkModel if fnWorkModel is not None else (lambda lvls: np.ones(len(lvls)))

        from itertools import count
        from collections import defaultdict
        self.alpha_dict = defaultdict(count(0).next)
        self.lvls = None

    def init_mimc_run(self, run):
        run.params.M0 = np.array([0])
        run.params.reuse_samples = False
        run.params.baysian = False
        run.params.moments = 1

    def update_index_set(self, lvls):
        if self.lvls is None:
            self.lvls = lvls
        assert(self.lvls == lvls)
        new_items = len(lvls) - len(self.alpha_ind)
        assert(new_items >= 0)
        new_alpha = lvls.sublist(np.arange(0, new_items) +
                                 len(self.alpha_ind)).to_dense_matrix(d_start=0, d_end=self.d)
        self.alpha_ind = np.hstack((self.alpha_ind,
                                    np.array([self.alpha_dict[tuple(k)] for
                                              k in new_alpha])))

    def sample_all(self, run, lvls, M, moments, fnSample):
        assert np.all(moments == 1), "miproj only support first moments"
        assert np.all(M == 1), "miproj only supports M=1 exactly"
        assert(self.lvls == lvls) # Assume the levels are the same
        assert(len(self.alpha_ind) == len(lvls))
        psums_delta = np.empty((len(lvls), 1), dtype=TensorExpansion)
        psums_fine = np.empty((len(lvls), 1), dtype=TensorExpansion)
        total_time = np.empty(len(lvls))
        total_work = np.empty(len(lvls))
        for alpha, ind in self.alpha_dict.iteritems():
            tStart = time.time()
            sel_lvls = self.alpha_ind == ind
            work_per_sample = self.fnWorkModel(setutil.VarSizeList([alpha]))
            beta_indset = lvls.sublist(sel_lvls, d_start=self.d, min_dim=0)
            max_dim = beta_indset.max_dim()
            basis = setutil.VarSizeList()
            pols_to_beta = []
            samples_per_beta = []
            for i, beta in enumerate(beta_indset):
                new_b, sam_b = self.fnBasisFromLvl(beta)
                samples_per_beta.append(sam_b)
                basis.add_from_list(new_b)
                pols_to_beta.extend([i]*len(new_b))

            pols_to_beta = np.array(pols_to_beta)
            samples_per_beta = np.array(samples_per_beta)
            c_samples = self.fnSamplesCount(basis)

            assert(np.all(basis.check_admissibility()))

            X, W = self.fnSamplePoints(c_samples, basis)
            basis_values = TensorExpansion.evaluate_basis(self.fnBasis, basis, X)
            mods, inds = mimc.expand_delta(alpha)
            for i in xrange(0, len(inds)):
                # Add each element separately
                Y = fnSample(inds[i], X)
                coeffs = MIWProjSampler.weighted_least_squares(Y, W, basis_values)
                projections = np.empty(len(beta_indset), dtype=TensorExpansion)

                for j in xrange(0, len(beta_indset)):
                    # if len(beta_indset[j]) == 0:
                    #     sel_coeff = np.ones(len(coeffs), dtype=np.bool)
                    # else:
                    sel_coeff = pols_to_beta == j
                    projections[j] = TensorExpansion(fnBasis=self.fnBasis,
                                                     base_indices=basis.sublist(sel_coeff),
                                                     coefficients=coeffs[sel_coeff])
                assert(np.all(np.sum(projections).coefficients == coeffs))
                assert(len(basis.set_diff(np.sum(projections).base_indices)) == 0)
                if i == 0:
                    psums_delta[sel_lvls, 0] = projections*mods[i]
                    psums_fine[sel_lvls, 0] = projections
                else:
                    psums_delta[sel_lvls, 0] += projections*mods[i]

            total_time[sel_lvls] = (time.time() - tStart) * samples_per_beta / c_samples
            total_work[sel_lvls] = work_per_sample * samples_per_beta

        return M, psums_delta, psums_fine, total_time, total_work

    @staticmethod
    def weighted_least_squares(Y, W, basisvalues):
        '''
        Solve least-squares system.

        :param Y: sample values
        :param W: weights
        :param basisvalues: polynomial basis values
        :return: coefficients
        '''
        R = basisvalues.transpose().dot(Y * W)
        G = basisvalues.transpose().dot(basisvalues * W[:, None])
        #coefficients = np.linalg.solve(G, R)
        if np.linalg.cond(G) > 100:
            warnings.warn('Ill conditioned Gramian matrix encountered')
        # Solving normal equations is faster than QR, because of good condition
        coefficients = solve(G, R, sym_pos=True)
        if not np.isfinite(coefficients).all():
            warnings.warn('Numerical instability encountered')
        return coefficients

def sample_uniform_pts(N, bases_indices, interval=(-1, 1)):
    dim = bases_indices.max_dim()
    return np.random.uniform(interval[0], interval[1], size=(N, dim)),\
        np.ones(N)*(1./N)

@public
def sample_optimal_pts(fnBasis, N, bases_indices, interval=(-1, 1)):
    max_dim = bases_indices.max_dim()
    acceptanceratio = 1./(4*np.exp(1))
    X = np.zeros((N, max_dim))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(N):
            pol = np.random.randint(0, len(bases_indices))
            base_pol = bases_indices.get_item(pol, max_dim)
            for dim in range(max_dim):
                accept=False
                while not accept:
                    Xnext = (np.cos(np.pi * np.random.rand()) + 1) / 2
                    dens_prop_Xnext = 1 / (np.pi * np.sqrt(Xnext*(1 - Xnext)))   # TODO: What happens if Xnext is 0
                    Xreal = interval[0] + Xnext *(interval[1] - interval[0])
                    dens_goal_Xnext = fnBasis(np.array([Xreal]), 1+base_pol[dim])[0,-1] ** 2
                    alpha = acceptanceratio * dens_goal_Xnext / dens_prop_Xnext
                    U = np.random.rand()
                    accept = (U < alpha)
                    if accept:
                        X[i,dim] = Xreal

    if X.shape[0] == 0:
        W = np.zeros(0)
    else:
        B = TensorExpansion.evaluate_basis(fnBasis, bases_indices, X)
        W = len(bases_indices) / np.sum(np.power(B, 2), axis=1)
    return X, W

@public
def default_basis_from_level(beta, C=2):
    # beta is zero indexed
    max_deg = 2 ** beta
    prev_deg = np.maximum(0, 2 ** (beta.astype(np.int)-1))
    l = len(beta)
    m = np.prod(max_deg)
    mprev = np.prod(prev_deg[prev_deg>0]) if np.any(prev_deg>0) else 0
    c_samples = (C * m * np.log2(m + 1)) - (C * mprev * np.log2(mprev + 1))
    return list(itertools.product(*[np.arange(prev_deg[i], max_deg[i])
                                    for i in xrange(0, l)])), c_samples


@public
def default_samples_count(basis, C=2):
    m = len(basis)
    return (4 if m == 1 else 0) + int(np.ceil(C * m * np.log2(m + 1)))

@public
def chebyshev_polynomials(Xtilde, N, interval=(-1,1)):
    r'''
    Compute values of the orthonormal Chebyshev polynomials on
    :math:`([-1,1],dx/2)` in :math:`X\subset [-1,1]`

    :param X: Locations of desired evaluations
    :param N: Number of polynomials
    :rtype: numpy.array of size :code:`X.shape[0]xN`
    '''
    X= (Xtilde-(interval[1]+interval[0])/2.)/((interval[1]-interval[0])/2.)

    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = np.concatenate((np.array([1]).reshape(1,1),np.sqrt(2)*np.ones((1,deg))),axis=1)
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = 2*X * out[:, n] - out[:, n - 1]
    return out * orthonormalizer

@public
def legendre_polynomials(Xtilde, N, interval=(-1,1)):
    r'''
    Compute values of the orthonormal Legendre polynomials on
    :math:`([-1,1],dx/2)` in :math:`X\subset [-1,1]`

    :param X: Locations of desired evaluations
    :param N: Number of polynomials
    :rtype: numpy.array of size :code:`X.shape[0]xN`
    '''
    X = (Xtilde-(interval[1]+interval[0])/2.)/((interval[1]-interval[0])/2.)
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = np.reshape(np.sqrt(2 * (np.array(range(deg + 1))) + 1), (1, N))
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = 1. / (n + 1) * ((2 * n + 1) * X * out[:, n] - n * out[:, n - 1])
    return out * orthonormalizer

@public
def hermite_polynomials(X, N):
    r'''
    Compute values of the orthonormal Hermite polynomials on
    :math:`(\mathbb{R},\frac{1}{\sqrt{2pi}}\exp(-x^2/2)dx)` in :math:`X\subset\mathbb{R}`


    :param X: Locations of desired evaluations
    :param N: Number of polynomials
    :rtype: numpy.array of size :code:`X.shape[0]xN`
    '''
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = 1/np.reshape([math.sqrt(math.factorial(n)) for n in range(N)], (1, N))
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = X * out[:, n] - n * out[:, n - 1]
    return out * orthonormalizer
