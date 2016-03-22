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
    dt = 1.0/32/(2**ell)
    return SZHW(dt,100.0,0.2,0.1,0.25,100.0,1.0,1.0,0.0,1.0,0.05,0.2,0.1,ell)  

# Compute the Black-Scholes option price using
# Multilevel Monte Carlo
# and solving a special case of
# the stochastic process in equation 2.1 of
# http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/Hybrid_SZHW.pdf

def MultiLevelBlackScholes(S,K,sig,T,r,TOL,Ca=2.0,estC=0.05,verbose=False,minSample=20):
    t1 = time.time()
    timestep = lambda ell: T/(2.0**ell)
    startL = 4
    muL = np.zeros(startL+1)
    muL[0] = BS0(S,sig,r,T,K)
    varL = np.zeros(startL+1)
    ML = [1] 
    for ell in range(1,startL+1):
        sample = np.array([BS(timestep(ell),S,sig,r,T,K) for foo in range(0,minSample)])
        muL[ell] = np.mean(sample)
        varL[ell] = np.var(sample)
        ML.append(minSample*1)
    L = 1*startL

    bias = abs(muL[L])
    var = varL[L]
    biasCoef = 0.5
    statCoef = 1.0-biasCoef
    maxVar = (TOL*statCoef/Ca)**2
    maxBias = biasCoef*TOL

    while ((var>maxVar) or (bias>maxBias)):
        if verbose:
            print('Starting an iteration, TOL=%f, stat error %f, bias %f'%(TOL,Ca*np.sqrt(var),bias))
        if bias > maxBias:
            if verbose:
                print('Bias at level %d is %f, reducing...'%(L,bias,))
            if np.sqrt(varL[L]/ML[L]) > 0.2*abs(muL[L]):
                if verbose:
                    print('Variance of bias estimator too large, Refining last level variance.')
                sample = np.array([BS(timestep(L),S,sig,r,T,K) for foo in range(0,ML[L])])
                muL[L] = 0.5*(np.mean(sample)+muL[L])
                varL[L] = 0.5*(np.var(sample)+varL[L])
                ML[-1] *= 2
                bias = abs(muL[L])+Ca*np.sqrt(varL[L]/ML[L])
            else:
                L += 1
                if verbose:
                    print('Added level.')
                ML.append(minSample)
                sample = np.array([BS(timestep(L),S,sig,r,T,K) for foo in range(0,minSample)])
                muL = np.concatenate((muL,np.mean(sample)*np.ones(1)))
                varL = np.concatenate((varL,np.var(sample)*np.ones(1)))
                bias = abs(muL[-1])+Ca*np.sqrt(varL[L]/ML[L])
        for ell in range(1,L+1):
            statisticalErrorBudget = maxVar/L
            levelVariance = varL[ell]/ML[ell]
            if (levelVariance>statisticalErrorBudget):
                if verbose:
                    print('For level %d variance is %f, %.2f %% of the budget.  Refining from M=%d to M=%d.'%(ell,levelVariance,levelVariance/statisticalErrorBudget*100,ML[ell],ML[ell]*2 ))
                sample = np.array([BS(timestep(ell),S,sig,r,T,K) for foo in range(0,ML[ell])])
                muL[ell] = 0.5*(np.mean(sample)+muL[ell])
                varL[ell] = 0.5*np.var(sample)+varL[ell]
                ML[ell] *= 2
        var = 0.0
        for ell in range(1,L+1):
            var += varL[ell]/ML[ell]
        bias = abs(muL[-1])++Ca*np.sqrt(varL[L]/ML[L])
    t2 = time.time()
    if verbose:
        print('Time elapsed %d seconds'%(t2-t1,))
    price = 0.0
    for ell in range(0,L+1):
        if verbose:
            print('Level %d, mu=%f (+/-) %f, M=%d'%(ell,muL[ell],Ca*np.sqrt(varL[ell]/ML[ell]),ML[ell]))
        price += muL[ell]

    if L>2:
        plt.figure()
        plt.semilogy(range(1,L+1),abs(muL[1:]),'r-',label='mean')
        plt.semilogy(range(1,L+1),[Ca*np.sqrt(varL[ell]/ML[ell]) for ell in range(1,L+1)],'b--',label='std. dev.')
        plt.xlabel('$l$')
        plt.legend()
        plt.ylabel('$\mu_l$')
        plt.grid(1)
        plt.title('BS multilevel,\n$S=%.2f$, $K=%.2f$, $r=%.2f$, $\sigma=%.2f$, $T=%0.2f$, $TOL=%.5f$'%(S,K,r,sig,T,TOL))
        plt.savefig('BSplot.pdf')

    return price




