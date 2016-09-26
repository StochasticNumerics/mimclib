from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import defaultdict
import itertools
from myutil import padldim
import set_util


class MISCData(object):
    def __init__(self, d, prevData=None, points_tol=1e-14, pooled=True):
        self.d = d
        self.pooled = pooled
        self.points_tol = points_tol
        if prevData is not None:
            assert(self.d == prevData.d)
            self.sample_pool = prevData.sample_pool
            self.knots_pool = prevData.knots_pool
        else:
            self.sample_pool = defaultdict(lambda: set_util.Tree())
            self.knots_pool = dict()

    def __solveAtPoints(self, sf, alpha, pts):
        # Samples (Y) points from the stochastic field (sf) using the mesh size
        # determined by (alpha). Assumes all Ys have the same size
        # Returns a vector of samples
        if not self.pooled:
            return sf(alpha, pts)
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
            new_values = sf(alpha, new_points)
            output[needsample] = new_values
            for i, pt in enumerate(new_points):
                pdict.add_node(pt, new_values[i], eps=self.points_tol)

        return output#, points

    def collapsePoints(self, pts):
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

    @staticmethod
    def tensor_from_pool(beta, knots_pool):
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
        m = [len(knots_pool[bj][0]) for bj in beta]
        sz = np.prod(m)
        knots = np.zeros((sz, N))
        weights = np.ones(sz)
        pattern = set_util.TensorGrid(m)
        for n in range(0, N):
            xx, ww = knots_pool[beta[n]]
            knots[:, n] = padldim(xx, 1)[pattern[:, n]-1]
            weights *= padldim(ww, 1)[pattern[:, n]-1]
        return knots.tolist(), weights

    def runMISC(self, sf, C, fnKnots, tensor_knots=False):
        # Assumes all variables have the same knots and lev2knots
        import time
        # First, we pool all knots, we don't want to include this calculation
        # in the timing.
        if tensor_knots:
            max_knots_count = int(np.max([np.max(ind[self.d:])
                                          if len(ind) > self.d else 0
                                          for ind in C]))
            for i in range(1, max_knots_count+1):
                if i in self.knots_pool:
                    continue
                self.knots_pool[i] = fnKnots(i)
            fnKnots = lambda beta: MISCData.tensor_from_pool(beta, self.knots_pool)

        error = np.zeros(len(C))
        work = np.zeros(len(C))
        for i, ind in enumerate(C):
            coeff, subind = expand_delta(ind)
            t = time.time()
            error[i] = 0
            for c, dind in zip(coeff, subind):
                alpha = dind[:self.d]
                beta = dind[self.d:]
                knots, weights = fnKnots(beta)
                knots = self.collapsePoints(knots)
                values = self.__solveAtPoints(sf, alpha, knots)
                error[i] += c*np.sum(weights * values)
            work[i] = time.time()-t

        return error, work


def run_MISC_indices(sf, sf_d,
                     C, knots, lev2knots, prev=None,
                     points_tol=1e-12, pooled=True,
                     fix_alpha=None, mlsc=False):
    # Runs MISC based on an index set, not optimization is done to reduce the
    # number samples, as such no advantage is taken of the nestedness of sample
    # points, if any.
    # (sf) is the stochastic field
    # (C) is the index set of dimension d+N
    # (knots) N-list of functions that return the knots for each  variable
    # (lev2knots) N-list of functions that return the number of points for every beta
    # (prev) Return value from a previous call to run_MISC,
    #              assumes that the current has been C expanded from the
    #              previous call
    # (pooled) if True, does not recompute samples
    #
    # Returns the "error" and "work" estimates for each index in C

    # C is the set of indices
    # knots is the function that returns n-points
    # lev2knots is what we call m(\beta)
    if fix_alpha is not None:
        assert(not mlsc)
        d = 0
    elif mlsc:
        d = 1
    else:
        d = sf_d

    miscData = MISCData(d, prev, points_tol, pooled=pooled)
    error, work = miscData.runMISC(sf, C,
                                   fnKnots = lambda beta: knots(lev2knots(beta)),
                                   tensor_knots = True)
    return error, work, miscData


################################################################
### FUNCTIONS BELOW ARE BETTER MOVED TO SPARSE_GRID SINCE THEY
### ARE GENERAL FOR ANY SPARSE GRID (EVEN MIMC)
################################################################
def get_profit_levels(profits, tol=1e-12):
    # Get a list of levels to construct a level sets
    # You can get a list of levels sets of profits by using:
    # [profits[:lvl] for lvl in get_profit_levels(profits)]
    # (tol) is the difference between profits such that the profits are equal
    # This function assumes profits are sorted least to most
    # Returns a list of indices of profits that define the boundary
    # of level sets.
    levels = []
    if len(profits) == 0:
        return np.array(levels, dtype=np.uint32)
    cur_profit = profits[0]
    for i, p in enumerate(profits[1:]):
        if np.isfinite(p) and p-cur_profit > tol:
            levels.append(i+1)
            cur_profit = p
    levels.append(profits.shape[0])
    return np.array(levels, dtype=np.uint32)


class MISCLevel:
    def __init__(self, profit, ind_diff, sel, inner_bnd, i, l, setSize):
        self.profit = profit
        self.ind_diff = ind_diff
        self.sel = sel
        self.inner_bnd = inner_bnd
        self.setSize = setSize
        self.i = i
        self.l = l

    def isBoundaryCalculated(self):
        return self.inner_bnd is not None
    def getBoundInd(self):
        # bnd_ind = np.zeros(self.setSize, dtype=np.bool)
        # inds = np.arange(0, self.inner_bnd.shape[0])
        # bnd_ind_sel = np.zeros(self.inner_bnd.shape[0], dtype=np.bool)
        # bnd_ind_sel[np.logical_and(self.inner_bnd >= self.i, inds < self.l)] = True
        # bnd_ind[self.sel] = bnd_ind_sel
        assert(self.isBoundaryCalculated())
        return set_util.GetBoundaryInd(self.setSize, self.inner_bnd,
                                       self.sel, self.l, self.i)

def prepare_set(U, Uprofits, excludeBoundary, realOnly=True,
                calcProf=None, d_admiss_start=0,
                computeBoundaries=False):
    # d_admiss_start: is the first dimension to check admissibility
    # Get only admissible indices
    import time
    sel = np.nonzero(U.CheckAdmissibility(d_start=d_admiss_start))[0]
    # Make admissible
    #profits_sel = Uprofits[sel]
    profits_sel = U.sublist(sel).MakeProfitsAdmissible(Uprofits[sel],
                                                       d_start=d_admiss_start)
    with np.errstate(invalid='ignore'):
        diffCount = np.sum(np.abs(profits_sel - Uprofits[sel]) > 0)
        if diffCount > 0:
            import warnings
            warnings.warn("Had to change {}/{} profits to make sets admissible".format(diffCount, len(sel)), RuntimeWarning)

    if excludeBoundary:
        if calcProf:
            min_profit = U.sublist(sel).calcMinOuterProf(calcProf)
        else:
            out_bnd, _ = set_util.GetAllBoundaries(U.sublist(sel))
            min_profit = np.min(profits_sel[out_bnd == 0])
        tmp = np.nonzero(profits_sel < min_profit)[0]
        sel, profits_sel = sel[tmp], profits_sel[tmp]

    tmp = np.argsort(profits_sel)
    sel, profits_sel = sel[tmp], profits_sel[tmp]
    lvls = get_profit_levels(profits_sel)
    if lvls.size == 0:
        return []  # No levels? Test this output

    if computeBoundaries:
        inner_bnd, real_lvls = set_util.GetAllBoundaries(U.sublist(sel), lvls)
    else:
        inner_bnd = None
        real_lvls = np.ones(len(lvls), dtype=np.bool)

    ll = np.concatenate((np.array([0], np.int), lvls))

    misc_lvls = []
    iprev = 0
    for i, l in enumerate(lvls):
        if realOnly and not real_lvls[i]:
            continue
        misc_lvl = MISCLevel(profits_sel[lvls[i]-1],
                             sel[ll[iprev]:ll[i+1]], sel, inner_bnd,
                             i, l, len(U))
        misc_lvls.append(misc_lvl)
        iprev = i+1
    return misc_lvls


def expand_delta(lvl, base=1):
     # Returns the indices that are contained in a certain (lvl)
     # Along with the modifiers the should multiply the value
     # For example, __expand_delta([2,3]) returns
     # [1,1,-1,-1], [[2,3], [1,2], [2,2], [1,3]]
     # or a permutation of these lists
    import itertools
    lvl = np.array(lvl, dtype=np.int)
    seeds = list()
    for i in range(0, lvl.shape[0]):
        if lvl[i] == base:
            seeds.append([0])
        else:
            seeds.append([0, 1])
    inds = np.array(list(itertools.product(*seeds)), dtype=np.int)
    mods = (2*np.sum(lvl) % 2 - 1) * (2*(np.sum(inds, axis=1) % 2) - 1)
    return mods, np.tile(lvl, (inds.shape[0], 1)) - inds


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

    if n == 1.:
        x = [(x_a+x_b)/2.]
        wt = [1.]
    else:
        # calculates the values of the recursive relation
        [a, b] = coeflege(n)
        # builds the matrix
        JacM = np.diag(a)+np.diag(np.sqrt(b[1:n]), 1.) + np.diag(
            np.sqrt(b[1:n]), (-1.))
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
    i = toarray(i, dtype=np.int);
    m = 2 ** (i-1)+1
    m[i==1] = 1
    m[i==0] = 0
    return m


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
