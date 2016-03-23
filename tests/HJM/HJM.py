#!/usr/bin/python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def hoLeeExample(inds,t_max=1.0,tau_max=2.0,r0=0.05,sig=0.01,verbose=False):
    
    '''
    Compute the Ho Lee Example in Beck-Tempone-Szepessy-Zouraris
    '''
    
    f0 = lambda tau: r0-sig*sig*0.5*tau*tau+0.1*(1.0-np.exp(-1*tau))

    if verbose:
        print('Evaluating the Ho Lee example.')
        print('r0: %f, vol %f , t_max %f , tau_max %f'%(r0,sig,t_max,tau_max))
        print('Evaluating with the following indices:')
        for ind in inds:
            print(ind)

    # largest values of the discretisation numbers
    N_t = max([foo[0] for foo in inds])
    N_tau_1 = max([foo[1] for foo in inds])
    N_tau_2 = max([foo[2] for foo in inds])

    N_t = 2**(N_t)+1
    N_tau_1 = 2**(N_tau_1)+1
    N_tau_2 = 2**(N_tau_2)+1

    if verbose:
        print('Meshes constructed.')
        print('The number of mesh points in time: %d'%(N_t))
        print('Mesh points in maturity: %d before t_max, %d after'%(N_tau_2,N_tau_1))

    times = np.linspace(0,t_max,N_t)
    taus_1 = np.linspace(0,t_max,N_tau_2)
    taus_2 = np.linspace(t_max,tau_max,N_tau_1)

    taus = np.concatenate((taus_1[:-1],taus_2))

    # initial values

    dt = times[1]-times[0]
<<<<<<< HEAD
    Ws = np.sqrt(dt)*sp.randn(N_t-1)
    plt.figure()
    Ws = np.concatenate((np.zeros(1),np.cumsum(Ws)))
    plt.plot(times,Ws)
=======
    Ws = np.concatenate((np.zeros(1),np.sqrt(dt)*np.cumsum(sp.randn(N_t-1))))
    if verbose:
        plt.figure()
        plt.plot(times,Ws)
        plt.xlabel('$t$')
        plt.ylabel('$W_t$')
        plt.grid(1)
>>>>>>> nohit

    rv = []
    
    for ind in inds:
        if verbose:
            print('Evaluating the following index:')
            print(ind)
        t_jump = 2**(max([foo[0] for foo in inds])-ind[0])
        tau_jump_1 = 2**(max([foo[1] for foo in inds])-ind[1])
        tau_jump_2 = 2**(max([foo[2] for foo in inds])-ind[2])
        if verbose:
            print('Jumps in each of the categories: %d , %d , %d'%(t_jump,tau_jump_1,tau_jump_2))
        tau_eff = np.concatenate((taus_1[0:-1:tau_jump_2],taus_2[0::tau_jump_1]))
        t_eff = times[::t_jump]
        f_eff = np.zeros((len(t_eff),len(tau_eff)+2))
        W_eff = Ws[0::t_jump]
        dt_eff = t_eff[1]-t_eff[0]
        if verbose:
            plt.figure()
            plt.plot(tau_eff,f_eff[0,:-2]+f0(tau_eff),'b-')
        # Time stepping
        lstar = 0
        for j in range(1,len(f_eff)):
            if verbose:
                print('Time step No %d , t=%f'%(j,t_eff[j]))
            f_eff[j,lstar:] = 1*f_eff[j-1,lstar:]
            f_eff[j,lstar:-2] += sig*sig*(tau_eff[lstar:]-t_eff[j-1])
            f_eff[j,lstar:-2] += sig*(W_eff[j]-W_eff[j-1])
            if verbose:
                plt.plot(tau_eff[lstar:],f_eff[j,lstar:-2]+f0(tau_eff[lstar:]),'b-')
            f_eff[j,-2] += f_eff[j-1,lstar]
            while tau_eff[lstar+1]<= t_eff[j]:
                lstar += 1
            f_eff[j,-2] = (f_eff[j-1,lstar]+f0(times[j-1]))*dt_eff
            # the last component unchanged
        if verbose:
            # plot the short rate
            lstar = 0
            tPlot = 1*t_eff
            fttPlot = 0*t_eff
            for j in range(0,len(f_eff)):
                fttPlot[j] = f_eff[j,lstar]
                while tau_eff[lstar+1]<= t_eff[j]:
                    lstar += 1
<<<<<<< HEAD
            plt.plot(tPlot,fttPlot+f0(tPlot),'r--')
            plt.xlabel('$\\tau$')
            plt.ylabel('$g(t,\\tau)$')
=======
            plt.plot(tPlot,fttPlot+f0(tPlot),'r-')
            plt.xlabel('$\\tau$')
            plt.ylabel('$f(t,\\tau)$')
>>>>>>> nohit
            plt.grid(1)

        rv.append(1.0-f_eff[-1,-2])
        tv = 0.0
        lstar = 0
        while tau_eff[lstar+1]<= t_max:
            lstar += 1
        rv[-1] *= np.sum(f_eff[-1,lstar:-3])*(tau_eff[-1]-tau_eff[-2])
        if verbose:
            print('The quantity of interest is %f'%(rv[-1]))
    
    return rv
    
