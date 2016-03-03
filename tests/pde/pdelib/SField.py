import numpy.ctypeslib as npct
import ctypes as ct
import numpy as np
import os

arr_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
arr_uint = npct.ndpointer(dtype=np.uint32, ndim=2, flags='CONTIGUOUS')


class SField(object):
    try:
        # TODO: We need to figure out a way
        save = ct.cdll._dlltype
        try:
            ct.cdll._dlltype = lambda name: ct.CDLL(name, ct.RTLD_GLOBAL)
            path = os.path.join(os.path.dirname(__file__))
            lib = npct.load_library("libsolver_nd_df.so", path)
        finally:
            ct.cdll._dlltype = save
        lib.SFieldCreate.restype = ct.c_ulong
        lib.SFieldBeginRuns.restype = ct.c_ulong
        lib.SFieldEndRuns.restype = ct.c_ulong
        lib.SFieldDestroy.restype = ct.c_ulong
        lib.SFieldSolveFor.restype = ct.c_ulong
        lib.SFieldGetDim.restype = ct.c_ulong
        lib.SFieldGetN.restype = ct.c_ulong

        lib.SFieldCreate.argtypes = [ct.c_voidp]
        lib.SFieldBeginRuns.argtypes = [ct.c_voidp, arr_double,
                                        arr_uint, ct.c_uint]
        lib.SFieldEndRuns.argtypes = [ct.c_voidp]
        lib.SFieldSolveFor.argtypes = [ct.c_voidp, arr_double,
                                       ct.c_uint, ct.c_void_p]
        lib.SFieldDestroy.argtypes = [ct.c_voidp]
        lib.SFieldGetDim.argtypes = [ct.c_voidp]
        lib.SFieldGetN.argtypes = [ct.c_voidp]
    except:
        raise
        pass

    def __init__(self, random_gen=None):
        self.random_gen = None or np.random
        self.ref = ct.c_voidp()
        self.checkErrCode(SField.lib.SFieldCreate(ct.byref(self.ref)))

    def GetDim(self):
        return SField.lib.SFieldGetDim(self.ref)

    def GetN(self):
        return SField.lib.SFieldGetN(self.ref)

    def BeginRuns(self, mods, nelem):
        # The following nelem insures nestedness
        mods = np.array(mods, dtype=np.double)
        qoi_dim = self.GetDim()
        nelem = nelem.astype(np.uint32)
        assert(nelem.shape[1] == 1 or nelem.shape[1] == qoi_dim)
        if nelem.shape[1] == 1 and qoi_dim != 1:
            nelem = np.tile(nelem, qoi_dim)
        self.checkErrCode(SField.lib.SFieldBeginRuns(self.ref,
                                                     mods, nelem,
                                                     nelem.shape[0]))

    def EndRuns(self):
        self.checkErrCode(SField.lib.SFieldEndRuns(self.ref))

    def __exit__(self, type, value, traceback):
        self.checkErrCode(SField.lib.SFieldDestroy(ct.byref(self.ref)))

    def __enter__(self):
        return self

    def SolveFor(self, Y):
        goal = ct.c_double()
        Y = np.array(Y)
        assert(Y.shape[0] == self.GetN())
        self.checkErrCode(SField.lib.SFieldSolveFor(self.ref,
                                                    Y,
                                                    Y.shape[0],
                                                    ct.byref(goal)))
        return goal.value

    def Sample(self):
        Y = self.random_gen.uniform(-np.sqrt(3), np.sqrt(3), size=self.GetN())
        return self.SolveFor(Y)

    def checkErrCode(self, errCode):
        if errCode == 0:
            return
        raise Exception("C code failed with {}".format(errCode))

    @staticmethod
    def Init():
        import sys
        count = len(sys.argv)
        arr = (ct.c_char_p * len(sys.argv))()
        arr[:] = sys.argv
        SField.lib.myPetscInit(count, arr)

    @staticmethod
    def Final():
        SField.lib.myPetscFinal()
