from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ctypes as ct
import numpy.ctypeslib as npct

__arr_double_1__ = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
__arr_uint32_1__ = npct.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')

import os
__lib__ = npct.load_library("_matern.so", __file__)
__lib__.SFieldCreate.restype = None
__lib__.SFieldCreate.argtypes = [ct.c_voidp, ct.c_int32,
                                 ct.c_double, ct.c_double,
                                 ct.c_uint32,
                                 ct.c_double, ct.c_double,
                                 ct.c_double, ct.c_double,
                                 ct.c_double, ct.c_double,
                                 __arr_double_1__, ct.c_double]
__lib__.SFieldBeginRuns.restype = None
__lib__.SFieldBeginRuns.argtypes = [ct.c_voidp, ct.c_uint32, __arr_uint32_1__]

__lib__.SFieldSolveFor.restype = ct.c_double
__lib__.SFieldSolveFor.argtypes = [ct.c_voidp, __arr_double_1__, ct.c_uint32]

__lib__.SFieldGetSolution.restype = None
__lib__.SFieldGetSolution.argtypes = [ct.c_voidp, __arr_double_1__,
                                      ct.c_uint32, __arr_double_1__,
                                      __arr_double_1__, ct.c_uint32]

# __lib__.SFieldEndRuns.restype = None
# __lib__.SFieldEndRuns.argtypes = [ct.c_voidp]

# __lib__.SFieldDestroy.restype = None
# __lib__.SFieldDestroy.argtypes = [ct.c_voidp]


class SField_Matern(object):
    def __init__(self, params):
        self.ref = ct.c_voidp()
        self.params = params
        assert(len(params.qoi_x0) >= params.qoi_dim), "x0 should have d dimension"
        #print("---------", d, a0, f0, nu, L, x0, sigma)
        example = 0
        if params.qoi_example == 'sf-matern':
            a, b = 0, 1
            example = 1
        elif params.qoi_example == 'sf-matern-full':
            a, b = 0, 1
            example = 0
        elif params.qoi_example == 'sf-kink':
            a, b = -1, 1
            example = 2
        else:
            raise ValueError('qoi_example is invalid')

        __lib__.SFieldCreate(ct.byref(self.ref),
                             example, a, b,
                             params.qoi_dim, params.qoi_a0,
                             params.qoi_f0, params.qoi_df_nu,
                             params.qoi_df_L, params.qoi_df_sig,
                             params.qoi_scale, params.qoi_x0,
                             params.qoi_sigma)


    def BeginRuns(self, ind, N):
        self.nelem = np.array(self.params.h0inv * self.params.beta**(np.array(ind)), dtype=np.uint32)
        if len(self.nelem) != self.GetDim():
            if len(self.nelem) == 1:
                # Just repeat it
                self.nelem = np.repeat(self.nelem, self.GetDim())
            else:
                assert(False)
        __lib__.SFieldBeginRuns(self.ref, N, self.nelem)

    def SolveFor(self, Y):
        return __lib__.SFieldSolveFor(self.ref, Y, Y.shape[0])

    def EndRuns(self):
        self.nelem = None
        __lib__.SFieldEndRuns(self.ref)

    def Sample(self, inds, M, rand_gen):
        sample_rand = rand_gen.uniform(-np.sqrt(3), np.sqrt(3),
                                       size=(M, self.GetN()))
        val = np.empty((M, len(inds)))
        for i, ind in enumerate(inds):
            self.BeginRuns(ind, self.GetN())
            for m in range(0, M):
                val[m, i] = self.SolveFor(sample_rand[m, :])
            self.EndRuns()
        return val

    def GetSolution(self, Y):
        size = np.prod(self.nelem)
        x = np.empty(int(self.GetDim() * size))
        y = np.empty(size)
        __lib__.SFieldGetSolution(self.ref, Y, len(Y),
                                  x, y, len(y))
        return x.reshape((self.GetDim(), size)), y

    def destroy(self):
        __lib__.SFieldDestroy(ct.byref(self.ref))

    def __exit__(self, type, value, traceback):
        self.destroy()
        return

    def __enter__(self):
        return self

    @staticmethod
    def Init():
        import sys
        count = len(sys.argv)
        arr = (ct.c_char_p * len(sys.argv))()
        arr[:] = sys.argv
        __lib__.myPetscInit(count, arr)

    @staticmethod
    def Final():
        __lib__.myPetscFinal()

    def GetDim(self):
        return self.params.qoi_dim

    def GetN(self):
        return self.params.qoi_N

if __name__=='__main__':
    from mimclib import Bunch
    params = Bunch()
    params.qoi_dim = 1
    params.qoi_a0 = 0
    params.qoi_f0 = 1
    params.qoi_df_nu = 3.5
    params.qoi_df_L = 1
    params.qoi_df_sig = 0.5
    params.qoi_scale = 1
    params.qoi_x0 = np.array([0.3,0.4, 0.6])
    params.qoi_sigma = 1
    params.h0inv = 3
    params.beta = 2

    arrY = np.array([[1,1,1,1], [1,1,1,1]], dtype=np.float)
    SField_Matern.Init()
    output = np.zeros(len(arrY))
    sf = SField_Matern(params)
    sf.BeginRuns(np.array([5]), np.max([len(Y) for Y in arrY]))
    for i, Y in enumerate(arrY):
        output[i] = sf.SolveFor(np.array(Y))
    sf.EndRuns()
    SField_Matern.Final()

    print(output)
