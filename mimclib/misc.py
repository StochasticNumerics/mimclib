from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from . import setutil
__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym

@public
class MISCSampler(object):
    def __init__(self, d, fnKnots, prevData=None, points_tol=1e-14, min_dim=0):
        self.d = d
        self.points_tol = points_tol
        self.fnKnots = fnKnots
        self.min_dim = min_dim
        if prevData is not None:
            assert(self.d == prevData.d)
            self.sample_pool = prevData.sample_pool
            self.knots_pool = prevData.knots_pool
        else:
            from collections import defaultdict
            self.sample_pool = defaultdict(lambda: setutil.Tree())
            self.knots_pool = dict()

    def _solveAtPoints(self, sf, alpha, pts):
        # Samples (Y) points from the stochastic field (sf) using the mesh size
        # determined by (alpha). Assumes all Ys have the same size
        # Returns a vector of samples
        if len(pts) == 0:
            return np.array([])
        N = len(pts[0])
        output = np.empty(len(pts))
        needsample = np.empty(len(pts), dtype=np.bool)
        pdict = self.sample_pool[tuple(alpha)]
        new_points = []
        for i, pt in enumerate(pts):
            val = pdict.find(pt, eps=self.points_tol)
            needsample[i] = val is None
            output[i] = val
            if needsample[i]:
                new_points.append(pt)

        if sum(needsample) > 0:
            new_values = sf(alpha, self.inflatePoints(new_points))
            output[needsample] = new_values
            for i, pt in enumerate(new_points):
                pdict.add_node(pt, new_values[i], eps=self.points_tol)
        return output#, points

    def collapsePoints(self, pts):
        # Remove zeros at the end of each point
        for i in range(0, len(pts)):
            mask = np.abs(pts[i]) > self.points_tol
            # get last
            try:
                last = (x for x in reversed([y for y in enumerate(mask)]) if x[1] == 1).next()[0]
                pts[i] = np.array(pts[i][:(last+1)])
            except StopIteration:
                pts[i] = np.array([])  # No points are greater than zero
                pass
        return pts

    def inflatePoints(self, pts):
        # Add zeros to the end of each point
        new_pts = []
        for i, pt in enumerate(pts):
            if len(pt) >= self.min_dim:
                new_pts.append(pt)
            else:
                new_pts.append(np.hstack((pt, np.zeros(self.min_dim-len(pt)))))
        return new_pts

    def tensor_from_pool(self, beta):
        # generate the pattern that will be used for knots and weights matrices, e.g.
        #
        # pattern = [1 1 1 1 2 2 2 2;
        #            1 1 2 2 1 1 2 2;
        #            1 2 1 2 1 2 1 2]
        #
        # meaning "first node d-dim uses node 1 in direction 1, 2 and 3, second d-dim  node uses node 1 in
        # direction 1 and 2 and node 2 in direction 3 ...
        # NOTE: The new C-implementation returns the transpose of the previous
        # structure for consistency with other functions in set_util

        # import itertools
        # pattern = np.array([s for s in itertools.product(*[range(1,j+1) for j in m])], dtype=np.int).transpose()
        # 1+np.rollaxis(np.indices(m), 0, len(m)+1).reshape(-1, len(m)).transpose()
        N = len(beta)
        if N == 0:
            return [[]], np.array([1.])
        m = [len(self.knots_pool[bj][0]) for bj in beta]
        sz = np.prod(m)
        knots = np.zeros((sz, N))
        weights = np.ones(sz)
        pattern = setutil.TensorGrid(m)

        def padldim(arr, d):
            arr = np.array(arr)
            return arr.reshape((1,)*(d-len(arr.shape)) + arr.shape)

        for n in range(0, N):
            xx, ww = self.knots_pool[beta[n]]
            knots[:, n] = padldim(xx, 1)[pattern[:, n]-1]
            weights *= padldim(ww, 1)[pattern[:, n]-1]
        return knots.tolist(), weights

    def update_knots_pool(self, inds):
        max_ind = [np.max(ind[self.d:]) for ind in inds if len(ind) > self.d]
        if len(max_ind) == 0:
            return
        max_knots_count = int(np.max(max_ind))
        for i in range(0, max_knots_count+1):
            if i in self.knots_pool:
                continue
            self.knots_pool[i] = self.fnKnots(i)

    def sample(self, inds, M, fnSample):
        assert M == 1, "MISC is a deterministic sampler, so M must be 1"
        self.update_knots_pool(inds)

        # TODO: Need to generalize to allow for array or general objects
        from . import mimc
        import time
        t = time.clock()
        samples = np.empty((M, len(inds)))
        for i, dind in enumerate(inds):
            alpha = dind[:self.d]
            beta = dind[self.d:]
            knots, weights = self.tensor_from_pool(beta)
            knots = self.collapsePoints(knots)
            values = self._solveAtPoints(fnSample, alpha, knots)
            samples[0, i] = np.sum(weights * values)
        work = time.clock()-t
        return samples, work

@public
def knots_gaussian(n, mi, sigma):
    # [x,w]=KNOTS_GAUSSIAN(n,mi,sigma)
    #
    # calculates the collocation points (x)
    # and the weights (w) for the gaussian integration
    # w.r.t to the weight function
    # rho(x)=1/sqrt(2*pi*sigma) *exp( -(x-mi)^2 / (2*sigma^2) )
    # i.e. the density of a gaussian random variable
    # with mean mi and standard deviation sigma
    # ----------------------------------------------------
    # Sparse Grid Matlab Kit
    # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
    # See LICENSE.txt for license
    # ----------------------------------------------------
    if n == 1:
        # the point (traslated if needed)
        # the weight is 1:
        return [mi], [1]

    def coefherm(n):
        if n <= 1:
            raise Exception(' n must be > 1 ')
        a = np.zeros(n)
        b = np.zeros(n)
        b[0] = np.sqrt(np.pi)
        b[1:] = 0.5 * np.arange(1, n)
        return a, b

    # calculates the values of the recursive relation
    a, b = coefherm(n)
    # builds the matrix
    JacM = np.diag(a)+np.diag(np.sqrt(b[1:n]), 1.)+np.diag(np.sqrt(b[1:n]), -1)
    # calculates points and weights from eigenvalues / eigenvectors of JacM
    [x, W] = np.linalg.eig(JacM)
    w = W[0, :]**2.
    ind = np.argsort(x)
    x = x[ind]
    w = w[ind]
    # modifies points according to mi, sigma (the weigths are unaffected)
    x = mi + np.sqrt(2) * sigma * x
    return x, w

@public
def knots_CC(nn, x_a, x_b, whichrho='prob'):
    # [x,w] = KNOTS_CC(nn,x_a,x_b)
    #
    # calculates the collocation points (x)
    # and the weights (w) for the Clenshaw-Curtis integration formula
    # w.r.t to the weight function rho(x)=1/(b-a)
    # i.e. the density of a uniform random variable
    # with range going from x=a to x=b.
    #
    # [x,w] = KNOTS_CC(nn,x_a,x_b,'prob')
    #
    # is the same as [x,w] = KNOTS_CC(nn,x_a,x_b) above
    #
    # [x,w]=[x,w] = KNOTS_CC(nn,x_a,x_b,'nonprob')
    #
    # calculates the collocation points (x)
    # and the weights (w) for the Clenshaw-Curtis integration formula
    # w.r.t to the weight function rho(x)=1
    # ----------------------------------------------------
    # Sparse Grid Matlab Kit
    # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
    # See LICENSE.txt for license
    # ----------------------------------------------------

    if nn == 1:
        x = np.array([(x_a+x_b)/2.])
        wt = np.array([1])
    elif nn % 2 == 0:
        raise Exception('error in knots_CC: Clenshaw-Curtis formula \n \
use only odd number of points')
    else:
        n = nn-1
        N = np.arange(1, n, 2)
        l = end_N = N.shape[0]
        m = n-l
        v0 = np.concatenate((2./N/(N-2.), [1./N[end_N-1]], np.zeros(m)))
        end_v0 = v0.shape[0]
        v2 = -v0[0:end_v0-1] - v0[end_v0-1:0:-1]

        g0 = -np.ones(n)
        g0[l] = g0[l]+n
        g0[m] = g0[m]+n
        g = g0/(n**2 - 1 + n % 2)

        wcc = np.real(np.fft.ifft(v2+g))
        wt = np.concatenate((wcc, [wcc[0]])) / 2.

        x = np.cos(np.arange(0, n+1) * np.pi / n)
        x = ((x_b-x_a)/2.)*x + (x_a+x_b)/2.

    if whichrho == 'nonprob':
        w = (x_b-x_a)*wt
    elif whichrho == 'prob':
        w = wt
    else:
        raise Exception('4th input not recognized')
    return x, w

@public
def knots_uniform(n, x_a, x_b, whichrho='prob'):
    # [x,w]=KNOTS_UNIFORM(n,x_a,x_b)
    #
    # calculates the collocation points (x)
    # and the weights (w) for the gaussian integration
    # w.r.t. to the weight function rho(x)=1/(b-a)
    # i.e. the density of a uniform random variable
    # with range going from x=a to x=b.
    #
    #
    # [x,w]=KNOTS_UNIFORM(n,x_a,x_b,'prob')
    #
    # is the same as [x,w]=KNOTS_UNIFORM(n,x_a,x_b) above
    #
    #
    # [x,w]=KNOTS_UNIFORM(n,x_a,x_b,'nonprob')
    #
    # calculates the collocation points (x)
    # and the weights (w) for the gaussian integration
    # w.r.t to the weight function rho(x)=1
    # ----------------------------------------------------
    # Sparse Grid Matlab Kit
    # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
    # See LICENSE.txt for license
    # ----------------------------------------------------

    def coeflege(n):
        if n <= 1:
            np.disp(' n must be > 1 ')
            return None

        a = np.zeros(n)
        b = np.zeros(n)
        b[0] = 2.
        k = np.arange(2, n+1)
        b[k-1] = 1. / (4. - 1. / (k-1)**2.)
        return [a, b]


    if n == 1.:
        x = [(x_a+x_b)/2.]
        wt = [1.]
    else:
        # calculates the values of the recursive relation
        [a, b] = coeflege(n)
        # builds the matrix
        JacM = np.diag(a)+np.diag(np.sqrt(b[1:n]), 1) + np.diag(np.sqrt(b[1:n]), -1)
        # calculates points and weights from eigenvalues / eigenvectors of JacM
        [x, W] = np.linalg.eig(JacM)
        wt = W[0, :]**2.
        ind = np.argsort(x)
        x = x[ind]
        # #ok<TRSRT>
        wt = wt[ind]
        # modifies points according to the distribution and its
        # interval x_a, x_b
        x = np.dot((x_b-x_a)/2., x)+(x_a+x_b)/2.

    # finally, fix weights
    if whichrho == 'nonprob':
        w = np.dot(x_b-x_a, wt)
    elif whichrho == 'prob':
        w = wt
    else:
        raise Exception('4th input not recognized')
    # ----------------------------------------------------------------------
    return x, w

@public
def lev2knots_doubling(i):
    # m = lev2knots_doubling(i)
    #
    # relation level / number of points:
    #    m = 2^{i-1}+1, for i>1
    #    m=1            for i=1
    #    m=0            for i=0
    #
    # i.e. m(i)=2*m(i-1)-1
    # ----------------------------------------------------
    # Sparse Grid Matlab Kit
    # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
    # See LICENSE.txt for license
    # ----------------------------------------------------
    scalar = np.isscalar(i)
    i = np.array([i] if scalar else i, dtype=np.int)
    m = 2 ** (i-1)+1
    m[i==1] = 1
    m[i==0] = 0
    if scalar:
        return m[0]
    return m

@public
def lev2knots_lin(i):
    #   relation level / number of points:
    #    m = i
    #
    #   [m] = lev2knots_lin(i)
    #   i: level in each direction
    #   m: number of points to be used in each direction
    #----------------------------------------------------
    # Sparse Grid Matlab Kit
    # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
    # See LICENSE.txt for license
    #----------------------------------------------------
    return i

@public
def estimate_misc_error_rates(d, lvls, errs,
                              lev2knots,
                              d_err_rates=None, tol=1e-14):
    errs = np.abs(errs)
    if tol is not None:
        sel = errs > tol
    else:
        sel = np.ones(len(lvls), dtype=np.bool)

    # Technically, the inactive dimensions should not be included in the fit
    if d_err_rates is not None:
        sel = np.logical_and(sel, lvls.get_dim() > d)

    lvls = lvls.sublist(sel)
    errs = errs[sel]

    if len(lvls) == 0:
        return d_err_rates, None   # Nothing to fit

    mat = lvls.to_dense_matrix()
    mat[:, d:] = lev2knots(mat[:, d:]-1)

    log_err = -np.log(errs)

    mat = np.hstack((mat, np.ones((mat.shape[0], 1))))
    from scipy.sparse import csr_matrix
    import scipy.sparse.linalg

    if d_err_rates is not None:
        log_err -= np.sum(mat[:, :d]*np.tile(d_err_rates, (mat.shape[0], 1)), axis=1)
        s_err_rates = scipy.sparse.linalg.lsqr(csr_matrix(mat[:, d:]), log_err)[0][:-1]
    else:
        raise NotImplementedError("This needs to be implemented as a non-linear optimization problem")
    return d_err_rates, s_err_rates
