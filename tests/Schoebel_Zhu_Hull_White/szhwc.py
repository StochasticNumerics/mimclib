#!/usr/bin/python

import ctypes
import numpy as np
import matplotlib.pyplot as plt
lib = ctypes.cdll.LoadLibrary('./libszhw.so')
lib.bsDt.argtypes = [ctypes.c_double]*6
lib.bsDt.restype = ctypes.c_double
lib.bsNoStep.restype = ctypes.c_double
lib.bsNoStep.argtypes = [ctypes.c_double]*5

# Function pointers to the C functions
BS = lib.bsDt 
BS0 = lib.bsNoStep

# Compute the Black-Scholes option price using
# Multilevel Monte Carlo
# and solving a special case of
# the stochastic process in equation 2.1 of
# http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/Hybrid_SZHW.pdf

def MultiLevelBlackScholes(S,K,sig,T,r,TOL,Ca=2.0):
    L = 1
    muL = np.zeros(L+1)
    muL[0] = BS0(S,sig,r,T,K)
    varL = np.zeros(L+1)
    minSize = 10
    ML = [1,minSize]
    sample = np.array([BS(T/2.0,S,sig,r,T,K) for foo in range(0,minSize)])
    muL[1] = np.mean(sample)
    varL[1] = np.sqrt(np.var(sample)/len(sample))
    bias = abs(muL[L])
    var = varL[L]
    maxVar = (TOL/2.0)**2
    maxBias = TOL/2.0
    while (sum(varL/np.array(ML))+bias > TOL):
        print('Starting an iteration, TOL=%f, stat error %f, bias %f'%(TOL,np.sqrt(sum(varL)),bias))
        if bias > maxBias:
            print('Bias is %f, reducing...'%(bias,))
            if np.sqrt(varL[L]/ML[L]) > 0.5*muL[L]:
                print('Bias is %f, Refining last level variance.'%(bias,))
                sample = np.array([BS(T/(2.0**L),S,sig,r,T,K) for foo in range(0,minSize)])
                muL[L] = 0.5*(np.mean(sample)+muL[L])
                varL[L] = 0.5*(np.var(sample)+varL[L])
                ML[-1] *= 2
                bias = abs(muL[L])+np.sqrt(varL[L]*Ca/ML[L])
            else:
                L += 1
                print('Added level. L=%d'%(L,))
                ML.append(minSize)
                sample = np.array([BS(T/(2.0**L),S,sig,r,T,K) for foo in range(0,minSize)])
                muL = np.concatenate((muL,np.mean(sample)*np.ones(1)))
                varL = np.concatenate((varL,np.var(sample)*np.ones(1)))
                bias = abs(muL[-1])+np.sqrt(varL[L]*Ca)
        for ell in range(1,L+1):
            if varL[ell]/ML[ell]>maxVar/L/Ca:
                print('Refining level %d by %d new elements'%(ell,ML[ell]))
                sample = np.array([BS(T/(2.0**ell),S,sig,r,T,K) for foo in range(0,minSize)])
                muL[ell] = 0.5*(np.mean(sample)+muL[ell])
                varL[ell] = 0.5*np.var(sample)+varL[ell]
                ML[ell] *= 2
    price = 0.0
    for ell in range(0,L+1):
        print('Level %d, mu=%f, relative variance=%f, M=%d'%(ell,muL[ell],varL[ell]/ML[ell]/muL[ell],ML[ell]))
        price += muL[ell]

    plt.figure()
    plt.semilogy(range(0,L+1),abs(muL),'r-',label='mean')
    plt.semilogy(range(1,L+1),abs(np.sqrt(varL[1:]/np.array(ML[1:]))),'b--',label='std. dev.')
    plt.xlabel('$l$')
    plt.legend()
    plt.ylabel('$\mu$')
    plt.grid(1)
    return price
