#!/usr/bin/python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.integrate as spint


def rateTest2D(Nref=6,M=100,r0=0.05,sig=0.01):
    
    """
    Test the convergence rates in two different dimensions.
    """

    # First check convergence along first difference

    plotX = range(1,Nref+1)
    plotYStrong = []
    plotYWeak = []
    plotYStrongE = []
    plotYWeakE = []

    plotRate = lambda ells,rate: 2**(np.array(ells)*(-1.0*rate))
    
    for ell in plotX:
        sample = [hoLeeExample2([[ell,Nref],[ell-1,Nref]]) for foo in range(M)]
        sample = [abs(foo[1]-foo[0]) for foo in sample]
        plotYWeak.append(np.mean(sample))
        plotYWeakE.append(np.sqrt(np.var(sample)/M))
        sample = [foo**2 for foo in sample]
        plotYStrong.append(np.mean(sample))
        plotYStrongE.append(np.sqrt(np.var(sample)/M))

    plotYStrong = np.array(plotYStrong)
    plotYStrongE = np.array(plotYStrongE)
    plotYWeak = np.array(plotYWeak)
    plotYWeakE = np.array(plotYWeakE)

    plt.figure()
    plt.semilogy(plotX,plotYWeak,'b-')
    plt.semilogy(plotX,plotYWeak-plotYWeakE,'b--')
    plt.semilogy(plotX,plotYWeak+plotYWeakE,'b--')
    plt.semilogy(plotX,0.1*plotRate(plotX,1.0),'r-')
    plt.grid(1)
    plt.title('Weak error, $\Delta t$ direction')
    plt.xlabel('$\ell$')
    plt.ylabel('$E_w$')
    plt.savefig('dim_1_weak.pdf')

    plt.figure()
    plt.semilogy(plotX,plotYStrong,'b-')
    plt.semilogy(plotX,plotYStrong+plotYStrongE,'b--')
    plt.semilogy(plotX,plotYStrong-plotYStrongE,'b--')
    plt.semilogy(plotX,1.0e-3*(plotRate(plotX,1.0))**2,'r-')
    plt.grid(1)
    plt.title('Strong error, $\Delta t$ dimension')
    plt.xlabel('$\ell$')
    plt.ylabel('$E_s$')
    plt.savefig('dim_1_strong.pdf')

    plotYStrong = []
    plotYWeak =[]
    plotYStrongE = []
    plotYWeakE = []

    for ell in plotX:
        sample = [hoLeeExample2([[Nref,ell],[Nref,ell-1]]) for foo in range(M)]
        sample = [abs(foo[1]-foo[0]) for foo in sample]
        plotYWeak.append(np.mean(sample))
        plotYWeakE.append(np.sqrt(np.var(sample)/M))
        sample = [foo**2 for foo in sample]
        plotYStrong.append(np.mean(sample))
        plotYStrongE.append(np.sqrt(np.var(sample)/M))

    plotYStrong = np.array(plotYStrong)
    plotYStrongE = np.array(plotYStrongE)
    plotYWeak =np.array(plotYWeak)
    plotYWeakE = np.array(plotYWeakE)

    plt.figure()
    plt.semilogy(plotX,plotYWeak,'b-')
    plt.semilogy(plotX,plotYWeak-plotYWeakE,'b--')
    plt.semilogy(plotX,plotYWeak+plotYWeakE,'b--')
    plt.semilogy(plotX,1*(plotRate(plotX,4.0)),'r-')
    plt.grid(1)
    plt.title('Weak error, $\Delta \\tau$ dimension')
    plt.xlabel('$\ell$')
    plt.ylabel('$E_w$')
    plt.savefig('dim_2_weak.pdf')

    plt.figure()
    plt.semilogy(plotX,plotYStrong,'b-')
    plt.semilogy(plotX,plotYStrong+plotYStrongE,'b--')
    plt.semilogy(plotX,plotYStrong-plotYStrongE,'b--')
    plt.semilogy(plotX,1*(plotRate(plotX,4.0))**2,'r-')
    plt.grid(1)
    plt.title('Strong error, $\Delta \\tau$ dimension')
    plt.xlabel('$\ell$')
    plt.ylabel('$E_s$')
    plt.savefig('dim_2_strong.pdf')



def hoLeeExample3(inds,t_max=1.0,tau_max=2.0,r0=0.05,sig=0.01,verbose=False):
    return hoLeeExample([[foo[0]]*3 for foo in inds],t_max=t_max,tau_max=tau_max,r0=r0,sig=sig,verbose=verbose)

def hoLeeExample2(inds,t_max=1.0,tau_max=2.0,r0=0.05,sig=0.01,verbose=False):
    return hoLeeExample([[foo[0],foo[1],foo[1]] for foo in inds],t_max=t_max,tau_max=tau_max,r0=r0,sig=sig,verbose=verbose)

def hoLeeExample(inds,t_max=1.0,tau_max=2.0,r0=0.05,sig=0.01,verbose=False):
    
    '''
    Compute the Ho Lee Example in Beck-Tempone-Szepessy-Zouraris
    '''
    
    thi = lambda tau: 0.1*(1-np.exp(-1*tau))
    f0 = lambda tau: r0-sig*sig*0.5*tau*tau+thi(tau)

    F = lambda x: 1.0-x
    G = lambda x: 1.0*x
    Psi = lambda x: 1.0*x
    U = lambda x: 0.0*x

    d1 = lambda s: sig*sig*s

    drift = lambda s: d1(s)

    v1 = lambda s: sig*np.ones(np.shape(s))

    vols = [v1,]

    identifierString = 'Evaluating the Ho Lee example.\n'
    identifierString += 'r0: %f, vol %f , t_max %f , tau_max %f .'

    return multiLevelHjmModel(inds,F,G,U,Psi,drift,vols,f0,t_max=t_max,tau_max=tau_max,identifierString=identifierString,verbose=verbose)
    
def twoFactorGaussianExample(inds,t_max=1.0,tau_max=3.0,b0=0.0759,b1=-0.0439,k=0.4454,a2=0.5,s1=0.02,s2=0.01,K=0.5,verbose=False):
    
    '''
    Compute the two factor Gaussian Example in Beck-Tempone-Szepessy-Zouraris
    '''
    

    f0 = lambda tau: b0+b1*np.exp(-1.0*k*tau)

    F = lambda x: np.exp(-1.0*x)
    G = lambda x: np.fmax(np.exp(-1.0*x)-K)
    Psi = lambda x: 1.0*x
    U = lambda x: 0.0*x
    
    d1 = lambda s: s1*s1*s
    d20 = lambda s: np.exp(-0.5*a2*s)
    d2 = lambda s: 2*s2*s2/a2*d20(s)*(1.0-d20(s))
    
    drift = lambda s: d1(s)+d2(s) 

    v1 = lambda s: s1*np.ones(np.shape(s))
    v2 = lambda s: s2*d20(s)

    vols = [v1,v2]

    identifierString = 'Evaluating the Two Factor Gaussian example.\n'
    identifierString += 's1: %f, s2: %f, b0: %f, tau_max: %f, t_max: %f\n'%(s1,s2,b0,tau_max,t_max)
    identifierString += 'k: %f, a2: %f, K: %f, b1: %f'%(k,a2,K,b1)

    return multiLevelHjmModel(inds,F,G,U,Psi,drift,vols,f0,t_max=t_max,tau_max=tau_max,identifierString=identifierString,verbose=verbose)    

def multiLevelHjmModel(inds,F,G,U,Psi,drift,vols,f0,t_max=1.0,tau_max=2.0,identifierString='HJM Model',verbose=False):
    
    '''
    Template to solve HJM type problems
    '''

    if verbose:
        print('Evaluating the Two Factor Gaussian example.')
        print(identifierString)
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
    
    dt = times[1]-times[0]
    Ws = []
    for foo in range(len(vols)):
        Ws.append(np.concatenate((np.zeros(1),np.sqrt(dt)*np.cumsum(sp.randn(N_t-1)))))

    Ws = np.array(Ws)

    if verbose:
        plt.figure()
        for foo in range(len(vols)):
            plt.plot(times,Ws[foo,:])
        plt.xlabel('$t$')
        plt.ylabel('$W_t$')
        plt.grid(1)

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
        Ws_eff = Ws[:,0::t_jump]
        dt_eff = t_eff[1]-t_eff[0]
        if verbose:
            plt.figure()
            plt.plot(tau_eff,f_eff[0,:-2]+f0(tau_eff),'g-')
        # Time stepping                                                                                                                                                                                              
        lstar = 0
        for j in range(1,len(f_eff)):
            if verbose:
                print('Time step No %d, t=%.4f. tau_n=%.4f'%(j,t_eff[j],tau_eff[lstar]))
            f_eff[j,lstar:] = f_eff[j-1,lstar:]
            f_eff[j,lstar:-2] += drift(tau_eff[lstar:]-t_eff[j-1])*dt_eff
            for foo in range(len(vols)):
                f_eff[j,lstar:-2] += vols[foo](tau_eff[lstar:]-t_eff[j-1])*(Ws_eff[foo,j]-Ws_eff[foo,j-1])
            while tau_eff[lstar+1]<= t_eff[j]:
                lstar += 1
            f_eff[j,-2] = f_eff[j-1,-2]+(f_eff[j-1,lstar]+f0(tau_eff[lstar]))*dt_eff
            f_eff[j,-1] = f_eff[j-1,-1]+(F(f_eff[j-1,-2])*U(f_eff[j-1,lstar]+f0(tau_eff[lstar])))*dt_eff
            while tau_eff[lstar+1]<= t_eff[j]:
                lstar += 1
            f_eff[j,-2] = f_eff[j-1,-2]+(f_eff[j-1,lstar]+f0(tau_eff[lstar]))*dt_eff
            f_eff[j,-1] = f_eff[j-1,-1]+(F(f_eff[j-1,-2])*U(f_eff[j-1,lstar]+f0(tau_eff[lstar])))*dt_eff
            if verbose:
                plt.plot(tau_eff[lstar:],f_eff[j,lstar:-2]+f0(tau_eff[lstar:]),'b-')
        if verbose:
            plt.plot(tau_eff[lstar:],f_eff[-1,lstar:-2]+f0(tau_eff[lstar:]),'r--')
            lstar = 0
            tPlot = 1*t_eff
            fttPlot = 0*t_eff
            for j in range(0,len(f_eff)):
                while tau_eff[lstar+1]<= t_eff[j]:
                    lstar += 1
                print('For t=%f, \\tau^*=%f'%(t_eff[j],tau_eff[lstar]))
                fttPlot[j] = f_eff[j,lstar]+f0(tau_eff[lstar])
                plt.plot([t_eff[j],],f_eff[j,lstar]+f0(tau_eff[lstar]),'gx')
            plt.plot(tPlot,fttPlot,'r-')
            plt.xlabel('$\\tau$')
            plt.ylabel('$f(t,\\tau)$')
            plt.grid(1)

        rv.append(F(f_eff[-1,-2]))
        if verbose:
            print('The discount term equals %f'%(rv[-1]))
        tv = 0.0
        lstar = 0
        while tau_eff[lstar+1]<= t_max:
            lstar += 1
        underlying = spint.simps(Psi(f_eff[-1,lstar:-2]+f0(tau_eff[lstar:])),tau_eff[lstar:])
        weirdTerm = f_eff[-1,-1]
        if verbose:
            print('The underlying term equals %f'%(underlying,))
            print('The absurd additive term equals %f'%(weirdTerm,))
        rv[-1] *= underlying
        rv[-1] += weirdTerm
        if verbose:
            print('The quantity of interest is %f'%(rv[-1]))

    return rv

