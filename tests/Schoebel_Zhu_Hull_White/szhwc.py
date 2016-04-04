#!/usr/bin/python

import ctypes
import numpy as np
import matplotlib.pyplot as plt
import time
lib = ctypes.cdll.LoadLibrary('./libszhw.so')
lib.bsDt.argtypes = [ctypes.c_double]*6
lib.bsDt.restype = ctypes.c_double
lib.bsNoStep.restype = ctypes.c_double
lib.bsNoStep.argtypes = [ctypes.c_double]*5
lib.test.argtypes = [ctypes.c_uint,ctypes.c_double]
lib.test.restype = ctypes.c_double
lib.szhwDt.restype = ctypes.c_double
lib.szhwDt.argtypes = [ctypes.c_double]*13 + [ctypes.c_bool]

# Function pointers to the C functions
BS = lib.bsDt 
BS0 = lib.bsNoStep
randTest = lib.test
SZHW = lib.szhwDt

def g_ell(ell):
    #double szhwDt(double dt, double S,double sigma, double r, double T, double K,double kappa,double lambda,double gamma,double p,double theta, double sbar, double eta,bool diff
    T = 1.0/12
    dt = T/4/(2**ell)
    return SZHW(dt,100.0,0.2,0.01,T,100.0,2.0,1.0,0.1,1.0,0.01,0.2,0.1,ell)

def wrap(run,inds):
    return g_ell(inds[0][0])
