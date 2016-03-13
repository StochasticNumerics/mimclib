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
__arr_bool__ = npct.ndpointer(dtype=np.bool, flags='C_CONTIGUOUS')
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

__lib__ = npct.load_library("_libset_util", __file__)

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
                                       __arr_double__, __arr_double__]

__lib__.FreeMemory.restype = None
__lib__.FreeMemory.argtypes = [ct.POINTER(ct.c_voidp)]

__lib__.GetMISCProfit.restype = ct.c_voidp
__lib__.GetMISCProfit.argtypes = [__ct_ind_t__, __ct_ind_t__,
                                  __arr_double__, __arr_double__,
                                  __arr_double__, __arr_double__]

__lib__.GetAnisoProfit.restype = ct.c_voidp
__lib__.GetAnisoProfit.argtypes = [__ct_ind_t__, __arr_double__, __arr_double__]

__lib__.FreeProfitCalculator.restype = None
__lib__.FreeProfitCalculator.argtype = [ct.c_voidp]

__lib__.FreeIndexSet.restype = None
__lib__.FreeIndexSet.argtype = [ct.c_voidp]

__lib__.GetIndexSet.restype = ct.c_voidp
__lib__.GetIndexSet.argtypes = [ct.c_voidp, ct.c_double,
                                ct.POINTER(ct.POINTER(ct.c_double))]
__lib__.GenTDSet.restype = None
__lib__.GenTDSet.argtypes = [__ct_ind_t__, __ct_ind_t__,
                             __arr_ind_t__, ct.c_uint32]

__lib__.TensorGrid.restype = None
__lib__.TensorGrid.argtypes = [__ct_ind_t__, __ct_ind_t__,
                               __arr_ind_t__, __arr_ind_t__, ct.c_uint32]


__lib__.VarSizeList_max_dim.restype = __ct_ind_t__
__lib__.VarSizeList_max_dim.argtypes = [ct.c_voidp]

__lib__.VarSizeList_get.restype = __ct_ind_t__
__lib__.VarSizeList_get.argtypes = [ct.c_voidp, ct.c_uint32, __arr_ind_t__,
                                    __ct_ind_t__]

__lib__.VarSizeList_count.restype = ct.c_uint32
__lib__.VarSizeList_count.argtypes = [ct.c_voidp]

__lib__.VarSizeList_sublist.restype = ct.c_voidp
__lib__.VarSizeList_sublist.argtypes = [ct.c_voidp, __arr_uint32__,
                                        ct.c_uint32]


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
__lib__.VarSizeList_from_matrix.argtypes = [__arr_ind_t__, ct.c_uint32,
                                            __arr_ind_t__, ct.c_uint32,
                                            __arr_ind_t__, ct.c_uint32]

__lib__.VarSizeList_find.restype = ct.c_int32
__lib__.VarSizeList_find.argtypes = [ct.c_voidp, __arr_ind_t__,
                                     __arr_ind_t__, __ct_ind_t__]


@public
class VarSizeList(object):
    def __init__(self, _handle, min_dim=0):
        self._handle = _handle
        self.min_dim = min_dim

    def __del__(self):
        __lib__.FreeIndexSet(self._handle)

    def __inflate_ind(self, ind):
        if len(ind) >= self.min_dim:
            return ind
        return np.concatenate((ind, np.ones(self.min_dim-len(ind),
                                            dtype=ind.dtype)))

    def sublist(self, sel):
        new = __lib__.VarSizeList_sublist(self._handle,
                                          np.array(sel, dtype=np.uint32),
                                          len(sel))
        return VarSizeList(new, min_dim=self.min_dim)

    def __getitem__(self, i):
        item = np.empty(np.maximum(self.min_dim, self.get_dim(i)), dtype=ind_t)
        __lib__.VarSizeList_get(self._handle, i, item, item.shape[0])
        return item

    def __iter__(self):
        def Iterate(self, dims):
            for i, dim in enumerate(np.arange(0, len(self))):
                item = np.empty(np.maximum(self.min_dim, dims[i]), dtype=ind_t)
                __lib__.VarSizeList_get(self._handle, i, item, item.shape[0])
                yield item
        return Iterate(self, self.get_dim())

    def __len__(self):
        return __lib__.VarSizeList_count(self._handle)

    def __str__(self):
        return "{ " + "\n  ".join([str(ind) for ind in self]) + " }"

    def max_dim(self):
        return __lib__.VarSizeList_max_dim(self._handle)

    def max_active_dim(self):
        return np.max(self.get_active_dim()) if len(self) > 0 else 0

    def to_matrix_base_0(self, d_start=0, d_end=None):
        d_end = d_end or self.max_dim()
        assert(d_end > d_start)
        ind_count = np.sum(self.get_active_dim())
        ij = np.empty(int(ind_count*2), dtype=ind_t)
        data = np.empty(ind_count, dtype=ind_t)
        __lib__.VarSizeList_to_matrix(self._handle, ij, len(ij), data,
                                      len(data))
        from scipy.sparse import csr_matrix
        mat = csr_matrix((data-1, (ij[::2], ij[1::2])),
                         shape=(len(self),
                                np.maximum(self.min_dim,
                                           np.maximum(d_end, self.max_dim()))))
        return mat[:, d_start:d_end]

    def to_dense_matrix(self, d_start=0, d_end=None, base=1):
        return np.array(self.to_matrix_base_0(d_start, d_end).todense()+base)

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

    def CheckAdmissibility(self, d_start=0, d_end=-1):
        if d_end < 0:
            d_end = self.max_dim()
        admissible = np.empty(len(self), dtype=np.bool)
        __lib__.CheckAdmissibility(self._handle, d_start, d_end,
                                   admissible)
        return admissible

    def MakeProfitsAdmissible(self, profits, d_start=0, d_end=-1):
        assert(len(profits) == len(self))
        if d_end < 0:
            d_end = self.max_dim()
        pro = profits.copy()
        __lib__.MakeProfitsAdmissible(self._handle, d_start, d_end, pro)
        return pro

    def find(self, ind):
        j_d = np.array([[i, j] for i, j in enumerate(ind) if j > 1],
                       dtype=ind_t).reshape((-1, 2))
        index = __lib__.VarSizeList_find(self._handle,
                                         j_d[:, 0].ravel(),
                                         j_d[:, 1].ravel(),
                                         j_d.shape[0])
        return index if index >= 0 else None

    @staticmethod
    def from_matrix(sizes, d_j, data, min_dim=0):
        assert(len(d_j) == len(data))
        assert(np.sum(sizes) == len(data))
        #print(np.array(data, dtype=ind_t))
        return VarSizeList(__lib__.VarSizeList_from_matrix(np.array(sizes, dtype=ind_t), len(sizes),
                                                           np.array(d_j, dtype=ind_t), len(d_j),
                                                           np.array(data, dtype=ind_t), len(data)),
                           min_dim=min_dim)

    def calcMinOuterProf(self, calcProf):
        return __lib__.GetMinOuterProfit(self._handle, calcProf._handle)

    def calcLogEW(self, profCalc):
        log_error = np.empty(len(self))
        log_work = np.empty(len(self))
        __lib__.CalculateSetProfit(self._handle, profCalc._handle,
                                   log_error, log_work)
        return log_error, log_work

    def calcLogProf(self, U):
        E, W = self.CalcLogEW(U)
        return W-E

    def GetAllBoundaries(C, lvls=None):
        if lvls is None:
            lvls = np.array([len(C)], dtype=np.uint32)
        else:
            lvls = np.array(lvls, dtype=np.uint32)

        inner_bnd = -1*np.ones(len(C), dtype=np.int32)
        real_lvls = np.zeros(len(lvls), dtype=np.bool)
        __lib__.GetLevelBoundaries(C._handle, lvls, len(lvls), inner_bnd, real_lvls)
        return inner_bnd, real_lvls


@public
class ProfCalculator(object):
    def GetIndexSet(self, max_prof):
        import ctypes as ct
        mem_prof = ct.POINTER(ct.c_double)()
        new = __lib__.GetIndexSet(self._handle,
                                  np.float(max_prof), ct.byref(mem_prof))
        indSet = VarSizeList(new, min_dim=self.d)
        try:
            count = len(indSet)
            profits = np.ctypeslib.as_array(mem_prof,
                                            (count,)).copy().reshape(count)
        finally:
            __lib__.FreeMemory(ct.byref(ct.cast(mem_prof, ct.c_void_p)))

        return indSet, profits

    def __del__(self):
        __lib__.FreeProfitCalculator(self._handle)


@public
class MISCProfCalculator(ProfCalculator):
    def __init__(self, d_err_rates, d_work_rates, s_g_rates,
                 s_g_bar_rates):
        assert(len(d_err_rates) == len(d_work_rates))
        assert(len(s_g_rates) == len(s_g_bar_rates))
        self.d = len(d_err_rates)
        self._handle = __lib__.GetMISCProfit(len(d_err_rates),
                                             len(s_g_rates),
                                             d_err_rates,
                                             d_work_rates, s_g_rates,
                                             s_g_bar_rates)


@public
class AnisoProfCalculator(ProfCalculator):
    def __init__(self, wE, wW):
        assert(len(wE) == len(wW))
        self.d = len(wE)
        self._handle = __lib__.GetAnisoProfit(self.d, wE, wW)


@public
def TensorGrid(m, base=1, count=None):
    m = np.array(m, dtype=ind_t)
    assert np.all(m >= base), "m has to be larger than base"
    count = count or np.prod(m-base+1)
    output = np.empty(count*len(m), dtype=ind_t)
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
