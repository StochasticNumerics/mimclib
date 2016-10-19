from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
import os.path

def python_wcumsum(x, w):
    assert(len(x) == w.shape[1])
    output = np.empty_like(w)
    for m in range(0, w.shape[0]):
        output[m, 0] = x[0]
        for i in range(1, w.shape[1]):
            output[m, i] = w[m, i]*output[m, i-1] + x[i]
    return output

try:
    # Try to import the DLL version of wcumsum,
    # This makes solving the SDE much faster
    import ctypes as ct
    import numpy.ctypeslib as npct
    __arr_double__ = npct.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')
    __libdir = os.path.join(os.path.dirname(__file__))
    __lib__ = npct.load_library("libwcumsum.so", __libdir)
    __lib__.wcumsum.restype = None
    __lib__.wcumsum.argtypes = [__arr_double__, __arr_double__,
                                ct.c_uint32, ct.c_uint32,
                                __arr_double__]

    def wcumsum(x, w):
        assert(len(x) == w.shape[1])
        output = np.empty_like(w)
        __lib__.wcumsum(x, w, w.shape[1], w.shape[0], output)
        return output

except:
    warnings.warn("Using Python's (very slow) version for wcumsum. Consider running make")
    # wcumsum is like cumsum, but weighted.
    wcumsum = python_wcumsum
