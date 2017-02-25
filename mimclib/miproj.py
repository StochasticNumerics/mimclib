from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from . import setutil
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
        # M Number of samples, dim is dimensions
        dim = np.max([len(x) for x in X])
        max_deg = np.max(base_indices.to_dense_matrix(), axis=0)
        rdim = np.minimum(dim, len(max_deg))
        values = np.ones((len(X), len(base_indices)))
        basis_values = np.empty(rdim, dtype=object)
        pt_dim = np.array([len(x) for x in X])
        for d in xrange(0, rdim):
            #vals = np.array([x[d] for x in X if d < len(x)])
            vals = np.array([x[d] if d < len(x) else 0 for x in X])
            #basis_values[d] = np.ones((len(X), max_deg[d]+1))
            #basis_values[d][pt_dim > d] = fnBasis(vals, max_deg[d]+1)
            basis_values[d] = fnBasis(vals, max_deg[d]+1)

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
    class SamplesCollection(object):
        def __init__(self):
            self.clear()

        def max_dim(self):
            return np.max([len(x) for x in self.X]) if len(self) > 0 else 0

        def clear(self):
            self.X = []
            self.W = np.empty(0)
            self.Y = None
            self.total_time = 0

        def add_points(self, fnSample, alphas, X, W):
            assert(len(X) == len(W))
            self.X.extend(X.tolist())
            self.W = np.hstack((self.W, W))
            if self.Y is None:
                self.Y = [np.zeros(0) for i in xrange(len(alphas))]
            assert(len(self.Y) == len(alphas))
            for i in xrange(0, len(alphas)):
                self.Y[i] = np.hstack((self.Y[i], fnSample(alphas[i], X)))

        @property
        def XWY(self):
            return self.X, self.W, self.Y

        def __len__(self):
            return len(self.X)

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

        self.prev_samples = defaultdict(lambda: MIWProjSampler.SamplesCollection())
        self.reuse_samples = reuse_samples
        self.max_condition_number = 0

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
            tStart = time.clock()
            sel_lvls = self.alpha_ind == ind
            work_per_sample = self.fnWorkModel(setutil.VarSizeList([alpha]))
            beta_indset = lvls.sublist(sel_lvls, d_start=self.d, min_dim=0)
            max_dim = beta_indset.max_dim()
            basis = setutil.VarSizeList()
            pols_to_beta = []
            basis_per_beta = []
            for i, beta in enumerate(beta_indset):
                new_b = self.fnBasisFromLvl(beta)
                basis_per_beta.append(len(new_b))
                basis.add_from_list(new_b)
                pols_to_beta.extend([i]*len(new_b))

            pols_to_beta = np.array(pols_to_beta)
            basis_per_beta = np.array(basis_per_beta)
            c_samples = self.fnSamplesCount(basis)

            assert(np.all(basis.check_admissibility()))
            mods, inds = mimc.expand_delta(alpha)

            if self.reuse_samples:
                if self.prev_samples[ind].max_dim() < basis.max_dim():
                    # Clear samples if dimensions are diff
                    self.prev_samples[ind].clear()

                if c_samples > len(self.prev_samples[ind]):
                    X, W = self.fnSamplePoints(c_samples - len(self.prev_samples[ind]), basis)
                    self.prev_samples[ind].add_points(fnSample, inds, X, W)
                X, W, Y = self.prev_samples[ind].XWY
            else:
                assert(c_samples > 0)
                X, W = self.fnSamplePoints(c_samples, basis)
                Y = [fnSample(inds[i], X) for i in xrange(0, len(inds))]

            sampling_time = time.clock() - tStart
            tStart = time.clock()
            basis_values = TensorExpansion.evaluate_basis(self.fnBasis, basis, X)
            assembly_time_1 = time.clock() - tStart

            tStart = time.clock()
            from scipy.sparse.linalg import gmres, LinearOperator
            BW = np.dot(np.diag(np.sqrt(W)), basis_values)
            G = LinearOperator((BW.shape[1], BW.shape[1]),
                               matvec=lambda v: np.dot(BW.transpose(), np.dot(BW, v)),
                               rmatvec=lambda v: np.dot(BW, np.dot(BW.transpose(), v)))
            assembly_time_2 = time.clock() - tStart

            # This following operation is only needed for diagnosis purposes
            GFull = basis_values.transpose().dot(basis_values * W[:, None])
            max_cond = np.linalg.cond(GFull)
            # assert np.max(np.abs(BW.transpose().dot(BW)-GFull)/GFull) < 1e-10, str(np.max(np.abs(BW.transpose().dot(BW)-GFull)/GFull))
            #G = GFull

            tStart = time.clock()
            for i in xrange(0, len(inds)):
                # Add each element separately
                R = np.dot(basis_values.transpose(), (Y[i] * W))
                coeffs, info = gmres(G, R)
                assert(info == 0)
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
            projection_time = time.clock() - tStart
            self.max_condition_number = np.maximum(self.max_condition_number,
                                                   max_cond)

            # For now, only compute sampling time
            time_taken = sampling_time + assembly_time_1 + \
                         assembly_time_2 + projection_time
            if self.reuse_samples:
                self.prev_samples[ind].total_time += time_taken
                time_taken = self.prev_samples[ind].total_time

            total_time[sel_lvls] = time_taken * basis_per_beta / np.sum(basis_per_beta)
            total_work[sel_lvls] = work_per_sample * c_samples * basis_per_beta / np.sum(basis_per_beta)
            ## WARNING: Not accounting for projection time!!!
            # print("{}, {}, {}, {}, {:.12f}, {:.12f}, {:.12f}, {:.12f}, {:.12f}"
            #       .format(len(basis), alpha[0], work_per_sample[0],
            #               c_samples,
            #               max_cond,
            #               sampling_time,
            #               assembly_time_1, assembly_time_2,
            #               projection_time))
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
        from scipy.linalg import solve
        R = basisvalues.transpose().dot(Y * W)
        G = basisvalues.transpose().dot(basisvalues * W[:, None])
        cond = np.linalg.cond(G)
        if cond > 100:
            warnings.warn('Ill conditioned Gramian matrix encountered, cond={}'.format(np.linalg.cond(G)))
        # Solving normal equations is faster than QR, because of good condition
        coefficients = solve(G, R, sym_pos=True)
        if not np.isfinite(coefficients).all():
            warnings.warn('Numerical instability encountered')
        return coefficients, cond


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
def sample_arcsine_pts(N, bases_indices, interval=(-1, 1)):
    max_dim = bases_indices.max_dim()
    X_temp = (np.cos(np.pi * np.random.rand(N, max_dim)) + 1) / 2
    X = interval[0]+X_temp*(interval[1]-interval[0])
    W = np.prod(np.pi * np.sqrt((X-interval[0])*(interval[1] - X)), axis=1)
    return (X,W)

@public
def default_basis_from_level(beta, C=2):
    # beta is zero indexed
    max_deg = 2 ** (beta + 1) - 1
    prev_deg = np.maximum(0, 2 ** beta - 1)
    l = len(beta)
    return list(itertools.product(*[np.arange(prev_deg[i], max_deg[i])
                                     for i in xrange(0, l)]))

@public
def default_samples_count(basis, C=2):
    m = len(basis)+1
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
