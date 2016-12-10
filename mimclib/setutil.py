from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import ctypes as ct
import numpy.ctypeslib as npct

__all__ = []


def public(sym):
    __all__.append(sym.__name__)
    return sym

__arr_int32__ = npct.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
__arr_uint32__ = npct.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS')
__arr_bool__ = npct.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS')
__arr_double__ = npct.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')

ind_t = np.uint16

__ct_ind_t__ = ct.c_uint16
__arr_ind_t__ = npct.ndpointer(dtype=ind_t, flags='C_CONTIGUOUS')

# def optimize_ind_size(ind, min_dim=0):
#     nz = np.nonzero(np.array(ind[::-1]) != 1)[0]
#     last = np.maximum(min_dim, 0 if len(nz) == 0 else (len(ind)-nz[0]))
#     return ind[:last]

# def optimize_set_size(C, min_dim=0):
#     newInd = []
#     sizes = np.zeros(len(C), dtype=np.uint32)
#     for i, ind in enumerate(C):
#         opt_ind = optimize_ind_size(ind, min_dim)
#         sizes[i] = len(opt_ind)
#         newInd.extend(opt_ind)
#     return VarSizeList(newInd, sizes=sizes)

__lib__ = npct.load_library("libset_util", __file__)

__lib__.CheckAdmissibility.restype = None
__lib__.CheckAdmissibility.argtypes = [ct.c_voidp, __ct_ind_t__,
                                       __ct_ind_t__, __arr_bool__]

__lib__.MakeProfitsAdmissible.restype = None
__lib__.MakeProfitsAdmissible.argtypes = [ct.c_voidp, __ct_ind_t__,
                                          __ct_ind_t__, __arr_double__]

# __lib__.GetLevelBoundaries.restype = None
# __lib__.GetLevelBoundaries.argtypes = [ct.c_voidp,
#                                        __arr_uint32__, ct.c_uint32,
#                                        __arr_int32__, __arr_bool__]

# __lib__.GetBoundaryInd.restype = None
# __lib__.GetBoundaryInd.argtypes = [ct.c_uint32, ct.c_uint32,
#                                    ct.c_int32, __arr_int32__,
#                                    __arr_int32__, __arr_bool__]

__lib__.GetMinOuterProfit.restype = ct.c_double
__lib__.GetMinOuterProfit.argtypes = [ct.c_voidp, ct.c_voidp]

__lib__.CalculateSetProfit.restype = None
__lib__.CalculateSetProfit.argtypes = [ct.c_voidp, ct.c_voidp,
                                       __arr_double__, ct.c_uint32]

__lib__.CreateMISCProfCalc.restype = ct.c_voidp
__lib__.CreateMISCProfCalc.argtypes = [__ct_ind_t__, __ct_ind_t__,
                                       __arr_double__, __arr_double__]

__lib__.CreateTDProfCalc.restype = ct.c_voidp
__lib__.CreateTDProfCalc.argtypes = [__ct_ind_t__, __arr_double__]

__lib__.CreateFTProfCalc.restype = ct.c_voidp
__lib__.CreateFTProfCalc.argtypes = [__ct_ind_t__, __arr_double__]

__lib__.FreeProfitCalculator.restype = None
__lib__.FreeProfitCalculator.argtypes = [ct.c_voidp]

__lib__.FreeIndexSet.restype = None
__lib__.FreeIndexSet.argtypes = [ct.c_voidp]

__lib__.GetIndexSet.restype = ct.c_voidp
__lib__.GetIndexSet.argtypes = [ct.c_voidp, ct.c_voidp, ct.c_double,
                                ct.POINTER(ct.POINTER(ct.c_double))]
__lib__.GenTDSet.restype = None
__lib__.GenTDSet.argtypes = [__ct_ind_t__, __ct_ind_t__,
                             __arr_ind_t__, ct.c_uint32]

__lib__.TensorGrid.restype = None
__lib__.TensorGrid.argtypes = [__ct_ind_t__, __ct_ind_t__,
                               __arr_ind_t__, __arr_ind_t__, ct.c_uint32]



__lib__.VarSizeList_count_neighbors.restype = None
__lib__.VarSizeList_count_neighbors.argtypes = [ct.c_voidp, __arr_ind_t__, ct.c_uint32]

__lib__.VarSizeList_is_parent_of_admissible.restype = None
__lib__.VarSizeList_is_parent_of_admissible.argtypes = [ct.c_voidp,
                                                        __arr_bool__, ct.c_uint32]

__lib__.VarSizeList_estimate_bias.restype = ct.c_double
__lib__.VarSizeList_estimate_bias.argtypes = [ct.c_voidp,
                                              __arr_double__, ct.c_uint32,
                                              __arr_double__, ct.c_uint32]

__lib__.VarSizeList_max_dim.restype = __ct_ind_t__
__lib__.VarSizeList_max_dim.argtypes = [ct.c_voidp]

__lib__.VarSizeList_get.restype = __ct_ind_t__
__lib__.VarSizeList_get.argtypes = [ct.c_voidp, ct.c_uint32, __arr_ind_t__,
                                    __arr_ind_t__, __ct_ind_t__]

__lib__.VarSizeList_count.restype = ct.c_uint32
__lib__.VarSizeList_count.argtypes = [ct.c_voidp]

__lib__.VarSizeList_sublist.restype = ct.c_voidp
__lib__.VarSizeList_sublist.argtypes = [ct.c_voidp,
                                        __ct_ind_t__, __ct_ind_t__,
                                        __arr_uint32__, ct.c_uint32]


__lib__.VarSizeList_all_dim.restype = None
__lib__.VarSizeList_all_dim.argtypes = [ct.c_voidp, __arr_uint32__,
                                        ct.c_uint32]

__lib__.VarSizeList_all_active_dim.restype = None
__lib__.VarSizeList_all_active_dim.argtypes = [ct.c_voidp, __arr_uint32__,
                                               ct.c_uint32]

__lib__.VarSizeList_get_dim.restype = ct.c_uint32
__lib__.VarSizeList_get_dim.argtypes = [ct.c_voidp, ct.c_uint32]

__lib__.VarSizeList_get_active_dim.restype = ct.c_uint32
__lib__.VarSizeList_get_active_dim.argtypes = [ct.c_voidp, ct.c_uint32]

__lib__.VarSizeList_to_matrix.restype = None
__lib__.VarSizeList_to_matrix.argtypes = [ct.c_voidp, __arr_ind_t__,
                                          ct.c_uint32, __arr_ind_t__,
                                          ct.c_uint32]

__lib__.VarSizeList_from_matrix.restype = ct.c_voidp
__lib__.VarSizeList_from_matrix.argtypes = [ct.c_voidp,
                                            __arr_ind_t__, ct.c_uint32,
                                            __arr_ind_t__, ct.c_uint32,
                                            __arr_ind_t__, ct.c_uint32]

__lib__.VarSizeList_find.restype = ct.c_int32
__lib__.VarSizeList_find.argtypes = [ct.c_voidp, __arr_ind_t__,
                                     __arr_ind_t__, __ct_ind_t__]

__lib__.VarSizeList_expand_set.restype = None
__lib__.VarSizeList_expand_set.argtypes = [ct.c_voidp, __arr_double__,
                                           __arr_double__,
                                           ct.c_uint32, __ct_ind_t__]
__lib__.VarSizeList_set_diff.restype = ct.c_voidp
__lib__.VarSizeList_set_diff.argtypes = [ct.c_voidp, ct.c_voidp]
__lib__.VarSizeList_set_union.restype = ct.c_voidp
__lib__.VarSizeList_set_union.argtypes = [ct.c_voidp, ct.c_voidp]

__lib__.VarSizeList_copy.restype = ct.c_voidp
__lib__.VarSizeList_copy.argtypes = [ct.c_voidp]

__lib__.VarSizeList_get_adaptive_order.restype = None
__lib__.VarSizeList_get_adaptive_order.argtypes = [ct.c_voidp,
                                                   __arr_double__,
                                                   __arr_double__,
                                                   __arr_uint32__,
                                                   ct.c_uint32,
                                                   __ct_ind_t__]

__lib__.Tree_new.restype = ct.c_voidp
__lib__.Tree_new.argtypes = []
__lib__.Tree_free.restype = None
__lib__.Tree_free.argtypes = [ct.c_voidp]

__lib__.Tree_add_node.restype = np.int8
__lib__.Tree_add_node.argtypes = [ct.c_voidp, __arr_double__,
                                  ct.c_uint32, ct.c_double,
                                  ct.c_double]

__lib__.Tree_find.restype = np.int8
__lib__.Tree_find.argtypes = [ct.c_voidp, __arr_double__,
                              ct.c_uint32, ct.POINTER(ct.c_double),
                              ct.c_uint32, ct.c_double]



__lib__.VarSizeList_check_errors.restype = None
__lib__.VarSizeList_check_errors.argtypes = [ct.c_voidp,
                                             __arr_double__,
                                             __arr_bool__,
                                             ct.c_uint32]


@public
class VarSizeList(object):
    def __init__(self, inds=None, **kwargs):
        self.min_dim = kwargs.pop("min_dim", 0)
        _handle = kwargs.pop("_handle", None)
        self._handle = None

        if _handle is None:
            self._handle = __lib__.VarSizeList_copy(0)
            if inds is not None:
                self.add_from_list(inds)
        else:
            assert inds is None, "Cannot set both _handle and inds"
            self._handle = _handle
        assert len(kwargs) == 0, "Unrecognized options {}".format(kwargs)

    # Pickle override
    def __getstate__(self):
        data, ind = self.to_list()
        return self.min_dim, data, ind

    def __setstate__(self, state):
        assert(not hasattr(self, "_handle"))
        self._handle = __lib__.VarSizeList_copy(0)
        self.min_dim = state[0]
        self.add_from_list(state[1], state[2])

    def copy(self):
        return VarSizeList(_handle=__lib__.VarSizeList_copy(self._handle),
                           min_dim=self.min_dim)

    def __del__(self):
        if self._handle is not None:
            __lib__.FreeIndexSet(self._handle)
            self._handle = None

    def sublist(self, sel=None, d_start=0, d_end=None):
        if sel is not None:
            sel = np.array(sel).reshape((-1,))
            if sel.dtype == np.bool:
                sel = np.nonzero(sel)[0]
            sel[sel < 0] = len(self) + sel[sel < 0]
            sel = sel.astype(dtype=np.uint32)
        if d_end is None:
            d_end = self.max_dim()
        new = __lib__.VarSizeList_sublist(self._handle,
                                          d_start, d_end,
                                          sel, len(sel))
        return VarSizeList(_handle=new, min_dim=self.min_dim)

    def get_item(self, i, dim=None):
        if dim is None:
            dim = np.maximum(self.min_dim, self.get_dim(i))
        item = np.empty(dim, dtype=ind_t)
        data = np.empty(self.get_active_dim(i), dtype=ind_t)
        j    = np.empty(len(data), dtype=ind_t)
        if i < 0:
            i = len(self) + i
        __lib__.VarSizeList_get(self._handle, i, data, j, len(data))
        item.fill(__lib__.GetDefaultSetBase())
        item[j] = data
        return item

    def __getitem__(self, i):
        return self.get_item(i)

    def __iter__(self):
        return self.dense_itr()

    def dense_itr(self, start=0, end=None):
        if end is None:
            end = len(self)
        dims = self.get_dim()
        active_dims = self.get_active_dim()
        for i in xrange(start, end):
            item = np.empty(np.maximum(self.min_dim, dims[i]), dtype=ind_t)
            data = np.empty(active_dims[i], dtype=ind_t)
            j    = np.empty(active_dims[i], dtype=ind_t)
            __lib__.VarSizeList_get(self._handle, i, data, j, len(data))
            item.fill(__lib__.GetDefaultSetBase())
            item[j] = data
            yield item

    def sparse_itr(self, start=0, end=None):
        if end is None:
            end = len(self)
        dims = self.get_active_dim()
        for i in xrange(start, end):
            data = np.empty(dims[i], dtype=ind_t)
            j    = np.empty(dims[i], dtype=ind_t)
            __lib__.VarSizeList_get(self._handle, i, data, j, len(data))
            yield j, data

    def __len__(self):
        return __lib__.VarSizeList_count(self._handle)

    def __str__(self):
        return "{ " + "\n  ".join([str(ind) for ind in self]) + " }"

    def max_dim(self):
        return __lib__.VarSizeList_max_dim(self._handle)

    def max_active_dim(self):
        return np.max(self.get_active_dim()) if len(self) > 0 else 0

    def to_list(self, d_start=0, d_end=None):
        d_end = d_end or np.maximum(1, self.max_dim())
        assert(d_end > d_start)
        sizes = self.get_active_dim()
        ind_count = np.sum(sizes)
        ij = np.empty(int(ind_count*2), dtype=ind_t)
        data = np.empty(ind_count, dtype=ind_t)
        __lib__.VarSizeList_to_matrix(self._handle, ij, len(ij), data,
                                      len(data))
        # Partition data based on sizes
        s = np.hstack((np.array([0], dtype=np.int), np.cumsum(sizes, dtype=np.int)))
        ind = ij[1::2]
        return [data[s[i]:s[i+1]] for i in xrange(0, len(s)-1)],\
            [ind[s[i]:s[i+1]] for i in xrange(0, len(s)-1)]

    def to_sparse_matrix(self, d_start=0, d_end=None):
        # Assumes that the martix is base 0
        d_end = d_end if d_end is not None else np.maximum(1, self.max_dim())
        assert(d_end >= d_start)
        ind_count = np.sum(self.get_active_dim())
        ij = np.empty(int(ind_count*2), dtype=ind_t)
        data = np.empty(ind_count, dtype=ind_t)
        __lib__.VarSizeList_to_matrix(self._handle, ij, len(ij), data,
                                      len(data))
        from scipy.sparse import csr_matrix
        mat = csr_matrix((data-__lib__.GetDefaultSetBase(), (ij[::2], ij[1::2])),
                         shape=(len(self),
                                np.maximum(self.min_dim,
                                           np.maximum(d_end, self.max_dim()))))
        return mat[:, d_start:d_end]

    def to_dense_matrix(self, d_start=0, d_end=None, base=0):
        return np.array(self.to_sparse_matrix(d_start, d_end).todense()+base)

    def get_dim(self, i=None):
        if i is None:
            dim = np.empty(len(self), dtype=np.uint32)
            __lib__.VarSizeList_all_dim(self._handle, dim, len(self))
            return dim
        return __lib__.VarSizeList_get_dim(self._handle, i)

    def get_active_dim(self, i=None):
        if i is None:
            dim = np.empty(len(self), dtype=np.uint32)
            __lib__.VarSizeList_all_active_dim(self._handle, dim, len(self))
            return dim
        return __lib__.VarSizeList_get_active_dim(self._handle, i)

    def check_admissibility(self, d_start=0, d_end=None):
        if d_end is None:
            d_end = self.max_dim()
        admissible = np.empty(len(self), dtype=np.int8)
        __lib__.CheckAdmissibility(self._handle, d_start, d_end, admissible)
        return admissible.astype(np.bool)

    def make_profits_admissible(self, profits, d_start=0, d_end=None):
        assert(len(profits) == len(self))
        if d_end is None:
            d_end = self.max_dim()
        pro = profits.copy()
        __lib__.MakeProfitsAdmissible(self._handle, d_start, d_end, pro)
        return pro

    def find(self, ind, j=None):
        if j is None:
            j = np.arange(0, len(ind), dtype=ind_t)
        else:
            j = np.array(j, dtype=ind_t)
        ind = np.array(ind, dtype=ind_t)
        index = __lib__.VarSizeList_find(self._handle, j, ind, len(ind))
        return index if index >= 0 else None

    def add_from_list(self, inds, j=None):
        sizes = np.array([len(a) for a in inds], dtype=ind_t)
        if j is None:
            j = [np.arange(0, len(i), dtype=ind_t) for i in inds]
        else:
            assert(len(j) == len(inds))
            assert np.all(sizes == np.array([len(a) for a in inds])), "Inconsistent data"
        j = np.hstack(j).astype(ind_t)
        inds = np.hstack(inds).astype(ind_t)
        __lib__.VarSizeList_from_matrix(self._handle,
                                        sizes, len(sizes),
                                        j, len(j), inds, len(inds))

    def calc_log_prof(self, U):
        log_prof = np.empty(len(self))
        __lib__.CalculateSetProfit(self._handle, profCalc._handle,
                                   log_prof, len(log_prof))
        return log_prof

    # def GetAllBoundaries(C, lvls=None):
    #     if lvls is None:
    #         lvls = np.array([len(C)], dtype=np.uint32)
    #     else:
    #         lvls = np.array(lvls, dtype=np.uint32)

    #     inner_bnd = -1*np.ones(len(C), dtype=np.int32)
    #     real_lvls = np.zeros(len(lvls), dtype=np.bool)
    #     __lib__.GetLevelBoundaries(C._handle, lvls, len(lvls), inner_bnd, real_lvls)
    #     return inner_bnd, real_lvls

    def set_diff(self, rhs):
        return VarSizeList(_handle=__lib__.VarSizeList_set_diff(self._handle, rhs._handle),
                           min_dim=self.min_dim)

    def set_union(self, rhs):
        return VarSizeList(_handle=__lib__.VarSizeList_set_union(self._handle, rhs._handle),
                           min_dim=self.min_dim)

    def get_adaptive_order(self, error, work, seedLookahead=5):
        assert(len(self) == len(error))
        assert(len(self) == len(work))
        adaptive_order = np.empty(len(self), dtype=np.uint32)
        __lib__.VarSizeList_get_adaptive_order(self._handle,
                                               error, work,
                                               adaptive_order,
                                               len(self),
                                               seedLookahead)
        return adaptive_order

    def check_errors(self, errors):
        assert(len(errors) == len(self))
        strange = np.empty(len(self), dtype=np.int8)
        __lib__.VarSizeList_check_errors(self._handle,
                                         errors,
                                         strange,
                                         len(errors))
        return strange.astype(np.bool)

    def count_neighbors(self):
        neigh = np.empty(len(self), dtype=ind_t)
        __lib__.VarSizeList_count_neighbors(self._handle,
                                            neigh, len(neigh))
        return neigh

    def is_parent_of_admissible(self):
        out = np.empty(len(self), dtype=np.int8)
        __lib__.VarSizeList_is_parent_of_admissible(self._handle, out, len(out))
        return out.astype(np.bool)

    def is_boundary(self):
        return self.count_neighbors() < self.max_dim()

    def expand_set(self, profCalc, max_prof=None):
        if max_prof is None:
            max_prof = self.get_min_outer_prof(profCalc)
        __lib__.GetIndexSet(self._handle, profCalc._handle, np.float(max_prof), None)

    def expand_set_adaptive(self, error, work, seedLookahead=5):
        assert(len(error) == len(self))
        assert(len(work) == len(self))
        return VarSizeList(_handle=__lib__.VarSizeList_expand_set(self._handle,
                                                          np.array(error, dtype=np.float),
                                                          np.array(work, dtype=np.float),
                                                          len(error),
                                                          seedLookahead),
                           min_dim=self.min_dim)


    def get_min_outer_prof(self, profCalc):
        return __lib__.GetMinOuterProfit(self._handle, profCalc._handle)

    def estimate_bias(self, err_contributions, rates=None):
        if rates is None:
            rates = np.ones(self.max_dim())
        return __lib__.VarSizeList_estimate_bias(self._handle,
                                                 err_contributions, len(err_contributions),
                                                 rates, len(rates))

class ProfCalculator(object):
    # def GetIndexSet(self, max_prof):
    #     import ctypes as ct
    #     mem_prof = ct.POINTER(ct.c_double)()
    #     new = __lib__.GetIndexSet(None, self._handle,
    #                               np.float(max_prof), ct.byref(mem_prof))
    #     indSet = VarSizeList(_handle=new, min_dim=self.d)
    #     try:
    #         count = len(indSet)
    #         profits = np.ctypeslib.as_array(mem_prof,
    #                                         (count,)).copy().reshape(count)
    #     finally:
    #         __lib__.FreeMemory(ct.byref(ct.cast(mem_prof, ct.c_void_p)))

    #     return indSet, profits
    def __del__(self):
        if self._handle is not None:
            __lib__.FreeProfitCalculator(self._handle)
            self._handle = None


class MISCProfCalculator(ProfCalculator):
    def __init__(self, d_rates, s_err_rates):
        self.d = len(d_rates)
        self._handle = __lib__.CreateMISCProfCalc(len(d_rates),
                                                  len(s_err_rates),
                                                  np.array(d_rates, dtype=np.float),
                                                  np.array(s_err_rates, dtype=np.float))

class TDProfCalculator(ProfCalculator):
    def __init__(self, w):
        self._handle = __lib__.CreateTDProfCalc(len(w), w)

class FTProfCalculator(ProfCalculator):
    def __init__(self, w):
        self._handle = __lib__.CreateFTProfCalc(len(w), w)

@public
def TensorGrid(m, base=1, count=None):
    m = np.array(m, dtype=ind_t)
    assert np.all(m >= base), "m has to be larger than base"
    count = count or np.prod(m-base+1)
    output = np.empty(int(count)*len(m), dtype=ind_t)
    __lib__.TensorGrid(len(m), base, m, output, count)
    return output.reshape((count, len(m)), order='C')


@public
def GenTDSet(d, count, base=1):
    output = np.empty(count*d, dtype=ind_t)
    __lib__.GenTDSet(d, base, output, count)
    return output.reshape((count, d), order='C')


# def GetBoundaryInd(setSize, inner_bnd, sel, l, i):
#     assert(len(sel) == len(inner_bnd))
#     assert(setSize >= len(inner_bnd))
#     bnd_ind = np.zeros(setSize, dtype=np.bool)
#     __lib__.GetBoundaryInd(len(inner_bnd), l, i, sel, inner_bnd,
#                            bnd_ind)
#     return bnd_ind


class Tree(object):
    def __init__(self, _handle=None):
        if _handle is None:
            self._handle = __lib__.Tree_new()
        else:
            self._handle = _handle

    def __del__(self):
        __lib__.Tree_free(self._handle)

    def add_node(self, value, data, eps=1e-14):
        value = np.array(value, dtype=np.float)
        prev_added = __lib__.Tree_add_node(self._handle, value, len(value), data, eps)
        assert(not prev_added)

    def find(self, value, eps=1e-14, remove=False):
        data = ct.c_double()
        value = np.array(value, dtype=np.float)
        found = __lib__.Tree_find(self._handle, value, len(value), ct.byref(data), remove, eps)
        if found:
            return data.value
        else:
            return None

    def output(self):
        __lib__.Tree_print(self._handle)
