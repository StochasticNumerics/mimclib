#!/usr/bin/python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.integrate as spint
import scipy.linalg as spal
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import itertools

def hash(x):
    return '%f_%f_%f_%f'%(x[0],x[-1],x[1]-x[0],x[-1]-x[-2])

def expCovar(xs,kappa,chol=False):
    '''
    Given a spatial mesh x, return an exponential
    covariance matrix with inverse correlation length
    kappa.
    '''

    x,y = np.meshgrid(xs,xs)
    rv = np.exp(-1.0*kappa*abs(x-y))
    if chol:
        return spal.cholesky(rv)
    return rv

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

def cCovExp(L,K,n):
    return 2*K*((1-np.exp(-abs(K*L))*((-1)**n))/(K*K+n*n*np.pi*np.pi/L/L))/L

def fourierExample(inds,cfun,L=50.0,tau1=1.0,tau2=2.0,t_max=1.0,verbose=False):

    ts = [time.time(),]

    w = lambda k: np.pi*k/L
    plotN = 200

    # largest values of the discretisation numbers                                                                           
    N_t = max([foo[0] for foo in inds])
    N_f = max([foo[1] for foo in inds])

    N_t = 2**(N_t)+1
    N_f = 2**(N_f)

    times = np.linspace(0,t_max,N_t)
    dt = times[1]-times[0]

    Wt = sp.randn(N_t,N_f)*np.sqrt(dt)
    Wt[0,:] *= 0
    Wt = np.cumsum(Wt,axis=0)

    if verbose:
        print('W mean and variance: %f, %f'%(np.mean(Wt[-1,:]),np.var(Wt[-1,:])))
    
    consts = np.zeros(N_t)
    ans = np.zeros((N_t,N_f-1))
    bns = np.zeros((N_t,N_f-1))

    for row in range(0,N_t):
        consts[row] = Wt[row,0]
        for col in range(0,N_f-1):
            k = col+1
            t = times[row]
            #ans[row,col] = ((cfun(k)/w(k))**2)*(np.cos(t*w(k))+np.sin(t*w(k))-1.0)
            ans[row,col] += cfun(k)*Wt[row,k]
            bns[row,col] = ans[row,col]
            #if (k%2==0):
            #    bns[row,col] -= (cfun(k)**2)*t/w(k)

    if verbose:
        plt.figure()
        plt.plot(times,ans[:,0],'b-')
        plt.plot(times,bns[:,0],'r-')
        plt.grid(1)
        

    zeroterm = lambda tt,tau: cfun(0)**2/2*tt*(tau-0.5*tt)

    if verbose:
        fig=plt.figure()
        ax=fig.gca(projection='3d')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$\\tau$')
        ax.set_zlabel('$f (\\tau, t)$')

        for row in range(0,N_t):
            t = times[row]
            plotx = np.linspace(t,tau2,int(plotN*(tau2)/(tau2-t)))
            ploty = zeroterm(t,plotx)
            ploty += consts[row]
            for k in range(1,N_f):
                col = k-1
                #print(row,col,ans[row,col],bns[row,col],w(k),consts[row])
                ploty += ans[row,col]*np.sin(w(k)*plotx)
                ploty += bns[row,col]*np.cos(w(k)*plotx)
            ax.plot(t*np.ones(np.shape(plotx)),plotx,ploty,'b-')

        shorts = []
        for row in range(0,N_t):
            shorts.append(consts[row])
            for k in range(1,N_f):
                col = k-1
                shorts[-1] += ans[row,col]*np.sin(w(k)*times[row])
                shorts[-1] += bns[row,col]*np.cos(w(k)*times[row])

        ax.plot(times,times,shorts,'r-')
    
    # solution of fourier coefficients done

    rv = []

    for ind in inds:
        if verbose:
            print('Evaluating the following index:')
            print(ind)
        t_jump = 2**(max([foo[0] for foo in inds])-ind[0])
        cutoff = 2**ind[1]
        rv2 = 0.0
        for k in range(1,cutoff):
            rv2 += bns[-1,k-1]*np.sin(tau2*w(k))
        #rv1 = ((cfun(0)**2)*times[-1]/2)*(tau2**2-tau2+tau1-tau1**2)
        #for k in range(1,cutoff):
            #rv1 += bns[-1,k-1]/w(k)*(np.sin(tau2*w(k))-np.sin(tau1*w(k)))
            #rv1 -= ans[-1,k-1]/w(k)*(np.cos(tau2*w(k))-np.cos(tau1*w(k)))
        #print('moi')
        #for ti in range(0,len(times)-1,t_jump):
            #t = times[ti]
            #rv2 += zeroterm(t,t)
            #for k in range(1,cutoff):
                #rv2 += ans[ti,k-1]*np.sin(t*w(k))
                #rv2 += bns[ti,k-1]*np.cos(t*w(k))
                #print('%d %d %f %f %f %f'%(ti,k,t,ans[ti,k-1],bns[ti,k-1],zeroterm(t,t)))
        rv2 *= dt*t_jump
        #if verbose:
        #    print('The two terms before exponentiation: %f and %f'%(rv1,rv2))
        #    print('Mean end value for W(s) %f'%(np.mean(Wt[-1,:])))
        #rv.append(np.exp(-1*rv1))
        #rv.append(np.exp(-1*rv2))
        rv.append(rv2)
        #rv.append(np.exp(-1*rv1)*np.exp(-1*rv2))

    return rv


def infDimHjmModel(inds,F,G,U,Psi,f0,kappa,t_max=1.0,tau_max=2.0,identifierString='Infinite HJM Model',verbose=False,maxLev=30):

    '''
    Solve an infinite-dimensional HJM model, crude example
    '''

    ts = [time.time(),]

    if verbose:
        print('Evaluating the Two Factor Gaussian example.')
        print(identifierString)
        for ind in inds:
            print(ind)

    # largest values of the discretisation numbers                                                                           
    N_t = max([foo[0] for foo in inds])
    N_tau_1 = max([foo[1] for foo in inds])
    N_tau_2 = max([foo[2] for foo in inds])

    if N_t+max(N_tau_1,N_tau_2) > maxLev:
        raise MemoryError('Asking for exceptionally refined solution!')


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
    tauhash = hash(taus)

    covMat = 1

    try:
        covMat = infDimHjmModel.chols[tauhash]
        if verbose:
            print('Cholesky factorisation already initialised.')
    except AttributeError:
        if verbose:
            print('Generating the cholesky factorisation.')
        infDimHjmModel.chols = {}
        covMat = expCovar(taus,kappa,chol=1)
        infDimHjmModel.chols[tauhash] = covMat
    except KeyError:
        if verbose:
            print('Generating the cholesky factorisation')
        covMat = expCovar(taus,kappa,chol=1)
        infDimHjmModel.chols[tauhash] = 1.0*covMat
        

    covMat = expCovar(taus,kappa,chol=1)

    dt = times[1]-times[0]
    Ws = np.zeros((len(times),len(taus)))
    for jj in range(1,len(Ws)):
        Ws[jj,:] = Ws[jj-1,:] + np.sqrt(dt)*np.dot(covMat,sp.randn(len(taus)))

    ts.append(time.time())

    if verbose:
        print('Mesh generations and initialisations done in %d seconds'%(ts[-1]-ts[-2]))

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
        cov_chol = expCovar(tau_eff,kappa,chol=1)
        t_eff = times[::t_jump]
        f_eff = np.zeros((len(t_eff),len(tau_eff)+2))
        Ws_eff = np.concatenate((Ws[0::t_jump,0:len(taus_1):tau_jump_1],Ws[0::t_jump,len(taus_1)::tau_jump_2]),axis=1)
        dt_eff = t_eff[1]-t_eff[0]
        if verbose:
            fig=plt.figure()
            ax=fig.gca(projection='3d')
            ax.plot(0*tau_eff,tau_eff,f_eff[0,:-2]+f0(tau_eff),'g-')
            ax.set_xlabel('$t$')
            ax.set_ylabel('$\\tau$')
            ax.set_zlabel('$f (\\tau, t)$')
        # Time stepping 
        lstar = 0
        for j in range(1,len(f_eff)):
            if verbose:
                print('Time step No %d, t=%.4f. tau_n=%.4f'%(j,t_eff[j],tau_eff[lstar]))
            f_eff[j,lstar:] = f_eff[j-1,lstar:]
            #f_eff[j,lstar:-2] += drift(tau_eff[lstar:]-t_eff[j-1])*dt_eff
            f_eff[j,lstar:-2] += Ws_eff[j,lstar:]-Ws_eff[j-1,lstar:]
            while tau_eff[lstar+1]<= t_eff[j]:
                lstar += 1
            f_eff[j,-2] = f_eff[j-1,-2]+(f_eff[j-1,lstar]+f0(tau_eff[lstar]))*dt_eff
            f_eff[j,-1] = f_eff[j-1,-1]+(F(f_eff[j-1,-2])*U(f_eff[j-1,lstar]+f0(tau_eff[lstar])))*dt_eff
            while tau_eff[lstar+1]<= t_eff[j]:
                lstar += 1
            f_eff[j,-2] = f_eff[j-1,-2]+(f_eff[j-1,lstar]+f0(tau_eff[lstar]))*dt_eff
            f_eff[j,-1] = f_eff[j-1,-1]+(F(f_eff[j-1,-2])*U(f_eff[j-1,lstar]+f0(tau_eff[lstar])))*dt_eff
            if verbose:
                ax.plot(t_eff[j]*np.ones(len(tau_eff[lstar:])),tau_eff[lstar:],f_eff[j,lstar:-2]+f0(tau_eff[lstar:]),'b-')
        if verbose:
            #ax.plot(tau_eff[lstar:],t_eff[j]*np.ones(np.shape(tau_eff[lstar:])),f_eff[-1,lstar:-2]+f0(tau_eff[lstar:]),'r--')
            lstar = 0
            tPlot = 1*t_eff
            fttPlot = 0*t_eff
            for j in range(0,len(f_eff)):
                while tau_eff[lstar+1]<= t_eff[j]:
                    lstar += 1
                print('For t=%f, \\tau^*=%f'%(t_eff[j],tau_eff[lstar]))
                fttPlot[j] = f_eff[j,lstar]+f0(tau_eff[lstar])
                ax.plot([t_eff[j],],[t_eff[j],],f_eff[j,lstar]+f0(tau_eff[lstar]),'gx')
            ax.plot(tPlot,tPlot,fttPlot,'r-')
            #plt.xlabel('$\\tau$')
            #plt.ylabel('$f(t,\\tau)$')
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
        ts.append(time.time())
        if verbose:
            print('The quantity of interest is %f'%(rv[-1]))
            print('Time spent on the level: %d seconds'%(ts[-1]-ts[-2]))

    if verbose:
        print('Total time for all inds: %d seconds'%(ts[-1]-ts[0]))

    return rv

def multiLevelHjmModel(inds,F,G,U,Psi,drift,vols,f0,t_max=1.0,tau_max=2.0,identifierString='HJM Model',verbose=False,maxLev=30):
    
    '''
    Template to solve HJM type problems
    '''

    ts = [time.time(),]

    if verbose:
        print('Evaluating the Two Factor Gaussian example.')
        print(identifierString)
        for ind in inds:
            print(ind)

    # largest values of the discretisation numbers
    N_t = max([foo[0] for foo in inds])
    N_tau_1 = max([foo[1] for foo in inds])
    N_tau_2 = max([foo[2] for foo in inds])

    if N_t+max(N_tau_1,N_tau_2) > maxLev:
        raise MemoryError('Asking for exceptionally refined solution!')
    

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

    ts.append(time.time())

    if verbose:
        print('Mesh generations and initialisations done in %d seconds'%(ts[-1]-ts[-2]))
    
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
                ax.plot(t_eff[j]*np.ones(len(tau_eff[lstar:])),tau_eff[lstar:],f_eff[j,lstar:-2]+f0(tau_eff[lstar:]),'b-')
        if verbose:
            #ax.plot(tau_eff[lstar:],f_eff[-1,lstar:-2]+f0(tau_eff[lstar:]),'r--')
            lstar = 0
            tPlot = 1*t_eff
            fttPlot = 0*t_eff
            for j in range(0,len(f_eff)):
                while tau_eff[lstar+1]<= t_eff[j]:
                    lstar += 1
                print('For t=%f, \\tau^*=%f'%(t_eff[j],tau_eff[lstar]))
                fttPlot[j] = f_eff[j,lstar]+f0(tau_eff[lstar])
                #ax.plot([t_eff[j],],f_eff[j,lstar]+f0(tau_eff[lstar]),'gx')
            #plt.plot(tPlot,fttPlot,'r-')
            #plt.xlabel('$\\tau$')
            #plt.ylabel('$f(t,\\tau)$')
            ax.set_xlabel('$t$')
            ax.set_ylabel('$\\tau$')
            ax.set_zlabel('$f (\tau,t)$')
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
        ts.append(time.time())
        if verbose:
            print('The quantity of interest is %f'%(rv[-1]))
            print('Time spent on the level: %d seconds'%(ts[-1]-ts[-2]))

    if verbose:
        print('Total time for all inds: %d seconds'%(ts[-1]-ts[0]))

    return rv

def hoLeeExample(inds,t_max=1.0,tau_max=2.0,r0=0.05,sig=0.01,verbose=False):
    
    '''
    Compute the Ho Lee Example in Beck-Tempone-Szepessy-Zouraris
    '''
    
    thi = lambda tau: 0.1*(1-np.exp(-1*tau))
    f0 = lambda tau: r0-sig*sig*0.5*tau*tau+thi(tau)

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
    Ws = np.concatenate((np.zeros(1),np.sqrt(dt)*np.cumsum(sp.randn(N_t-1))))
    if verbose:
        plt.figure()
        plt.plot(times,Ws)
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
        W_eff = Ws[0::t_jump]
        dt_eff = t_eff[1]-t_eff[0]
        if verbose:
            plt.figure()
            plt.plot(tau_eff,f_eff[0,:-2]+f0(tau_eff),'r-')
        # Time stepping
        lstar = 0
        for j in range(1,len(f_eff)):
            if verbose:
                print('Time step No %d, t=%.4f. tau_n=%.4f'%(j,t_eff[j],tau_eff[lstar]))
                #print('Time step No %d , t=%f'%(j,t_eff[j]))
            f_eff[j,lstar:] = 1*f_eff[j-1,lstar:]
            f_eff[j,lstar:-2] += sig*sig*(tau_eff[lstar:]-t_eff[j-1])*dt_eff
            f_eff[j,lstar:-2] += sig*(W_eff[j]-W_eff[j-1])
            if verbose:
                plt.plot(tau_eff[lstar:],f_eff[j,lstar:-2]+f0(tau_eff[lstar:]),'b-')
            f_eff[j,-2] += f_eff[j-1,lstar]
            while tau_eff[lstar+1]<= t_eff[j]:
                lstar += 1
            f_eff[j,-2] = (f_eff[j-1,lstar]+f0(times[j-1]))*dt_eff
            # the last component unchanged
        if verbose:
            plt.plot(tau_eff[lstar:],f_eff[-1,lstar:-2]+f0(tau_eff[lstar:]),'r--')
            plt.plot(tau_eff[lstar:],r0-0.5*sig*sig*(tau_eff[lstar:]-t_max)**2+thi(tau_eff[lstar:]),'k-.')
            # plot the short rate
            lstar = 0
            tPlot = 1*t_eff
            fttPlot = 0*t_eff
            for j in range(0,len(f_eff)):
                fttPlot[j] = f_eff[j,lstar]+f0(0.0)
                while tau_eff[lstar+1]<= t_eff[j]:
                    lstar += 1
            plt.plot(tPlot,fttPlot+f0(tPlot),'r-')
            plt.xlabel('$\\tau$')
            plt.ylabel('$f(t,\\tau)$')
            plt.grid(1)

        rv.append(1.0-f_eff[-1,-2])
        if verbose:
            print('The discount term equals %f'%(rv[-1]))
        tv = 0.0
        lstar = 0
        while tau_eff[lstar+1]<= t_max:
            lstar += 1
        underlying = spint.simps(f_eff[-1,lstar:-2]+f0(tau_eff[lstar:]),tau_eff[lstar:])
        if verbose:
            print('The underlying term equals %f'%(underlying,))
            #print('dtau term %f'%((tau_eff[-1]-tau_eff[-2])))
            #print('average forward curve %f'%(np.mean(f_eff[-1,lstar:-3])))
        rv[-1] *= underlying
        if verbose:
            print('The quantity of interest is %f'%(rv[-1]))
    
    return rv

def infExample2(inds):
    inp = [[foo[0],foo[1],foo[1]] for foo in inds]
    return infDimTest(inp)

def rateTest2D(fun=infExample2,Nref=7,M=100,r0=0.05,sig=0.01,weaks=[1,1],strongs=[1,1]):
    
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
        sample = [fun([[ell,Nref],[ell-1,Nref]]) for foo in range(M)]
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
    plt.semilogy(plotX,0.1*plotRate(plotX,weaks[0]*1.0),'r-')
    plt.grid(1)
    plt.title('Weak error, $\Delta t$ direction')
    plt.xlabel('$\ell$')
    plt.ylabel('$E_w$')
    plt.savefig('dim_1_weak.pdf')

    plt.figure()
    plt.semilogy(plotX,plotYStrong,'b-')
    plt.semilogy(plotX,plotYStrong+plotYStrongE,'b--')
    plt.semilogy(plotX,plotYStrong-plotYStrongE,'b--')
    plt.semilogy(plotX,1.0e-3*(plotRate(plotX,strongs[0]*1.0))**2,'r-')
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
        sample = [fun([[Nref,ell],[Nref,ell-1]]) for foo in range(M)]
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
    plt.semilogy(plotX,1*(plotRate(plotX,weaks[1]*1.0)),'r-')
    plt.grid(1)
    plt.title('Weak error, $\Delta \\tau$ dimension')
    plt.xlabel('$\ell$')
    plt.ylabel('$E_w$')
    plt.savefig('dim_2_weak.pdf')

    plt.figure()
    plt.semilogy(plotX,plotYStrong,'b-')
    plt.semilogy(plotX,plotYStrong+plotYStrongE,'b--')
    plt.semilogy(plotX,plotYStrong-plotYStrongE,'b--')
    plt.semilogy(plotX,1*(plotRate(plotX,strongs[1]*1.0))**2,'r-')
    plt.grid(1)
    plt.title('Strong error, $\Delta \\tau$ dimension')
    plt.xlabel('$\ell$')
    plt.ylabel('$E_s$')
    plt.savefig('dim_2_strong.pdf')

def infDimTest(inds,verbose=False,maxLev=20):
    F =lambda x: 1.0-x
    G =lambda x: 1.0*x
    U =lambda x: 0.0*x
    Psi = lambda x: 1.0*x
    f0 = lambda x: 0.0*x
    kappa = 2.0
    identifierString = 'infDim example, kappa=%f'%(kappa,)
    return infDimHjmModel(inds,F,G,U,Psi,f0,kappa,maxLev=maxLev,verbose=verbose)


def testcude(inds,L=10.0,tau1=1.0,tau2=2.0,verbose=False):
    c = lambda k: np.sqrt(0.5*cCovExp(L,10.0,k)) # fourier term for covariance
    w = lambda k: np.pi*k/L # frequency k
    sk = lambda k,x: np.sin(w(k)*x) # sine basis function
    ck = lambda k,x: np.cos(w(k)*x) # cosine basis function
    ski = lambda k,t1,t2: (ck(k,t1)-ck(k,t2))/w(k)
    cki = lambda k,t1,t2: (sk(k,t2)-sk(k,t1))/w(k)
    zt = lambda t,tau: (c(0)**2)*(tau*t-0.5*t*t) # zero term
    zi = lambda T,t1,t2: (c(0)**2)*((t2**2-t1**2)*T*0.5-T*T*(t2-t2)) #integral over zero term
    ntimes = max([ind[0] for ind in inds])
    ntimes = 2**ntimes+1
    times = np.linspace(0,1,ntimes)
    dt = times[1]-times[0]
    nmodes = max([ind[1] for ind in inds])
    nmodes = 2**nmodes
    Ws = sp.randn(len(times),nmodes)
    Ws *= np.sqrt(dt)
    Ws[0,:] *=0
    Ws = np.cumsum(Ws,axis=0)
    Wfinal = 1*Ws[-1,1:]
    ans = 0*Ws[:,1:]
    bns = 0*Ws[:,1:]
    zns = 0*Ws[:,0]
    afinal = 0*Wfinal
    bfinal = 0*Wfinal
    zfinal = 0

    '''
    for col in range(len(ans[0])):
        k = col+1
        for row in range(len(ans)):
            t = times[row]
            ans[row,col] += ((c(k)/w(k))**2 )*(ck(k,t)+sk(k,t)-1.0)
            bns[row,col] += ((c(k)/w(k))**2 )*(ck(k,t)+sk(k,t)-1.0)
            if not k%2:
                ans[row,col] -= c(k)*c(k)*t/w(k)
                bns[row,col] -=c(k)*c(k)*t/w(k)
    '''

    # drift part done
    for col in range(len(ans[0])):
        k = col+1
        for row in range(len(ans)):
            t = times[row]
            ans[row,col] += c(k)*Ws[row,col+1]
            bns[row,col] += c(k)*Ws[row,col+1]

    # stochastic bit done
    for col in range(len(ans[0])):
            k = col+1
            for row in range(len(ans)):
                t =times[row]
                ans[row,col] *= sk(k,t)
                bns[row,col] *= ck(k,t)

    # t=tau set
    for row in range(len(ans)):
        t = times[row]
        zns[row] = Ws[row,0]*c(0)
        zns[row] += zt(t,t)
    for col in range(len(ans[0])):
        k = col + 1
        T = times[-1]
        afinal[col] *= c(k)*Wfinal[col]
        bfinal[col] *= c(k)*Wfinal[col]
        afinal[col] += ((c(k)/w(k))**2)*(ck(k,T)+sk(k,T)-1.0)
        bfinal[col] += ((c(k)/w(k))**2)*(ck(k,T)+sk(k,T)-1.0)
        if not k%2:
            afinal[col] -=c(k)*c(k)*T/w(k)
            bfinal[col] -=c(k)*c(k)*T/w(k)
        afinal *= ski(k,tau1,tau2)
        bfinal *= cki(k,tau1,tau2)
    # integrals over the underlyings computed
    rv1 = []
    for ind in inds:
        jump = 2**(max([foo[0] for foo in inds]) - ind[0])
        cutoff = 2**(ind[1])
        summand = ans[0:-1:jump,:cutoff]+bns[0:-1:jump,:cutoff]
        rv1.append((dt*jump)*np.sum(summand))
        #rv1[-1] += dt*jump*np.sum(zns[:-1])
        #rv1[-1] = np.exp(-1*rv1[-1])
    rv2 = []
    for ind in inds:
        cutoff = 2**(ind[1])
        rv2.append(np.sum(Wfinal[:cutoff]))
        rv2[-1] += zi(times[-1],tau1,tau2)
    #return [np.exp(-1*rv1[foo])*np.exp(-1*rv2[foo]) for foo in range(len(inds))]
    #return rv2
    return [np.exp(-1*foo) for foo in rv1]

def aTest(ks):
    times = np.linspace(0,1,100)
    dt = times[1]-times[0]
    W = sp.randn(100)
    W[0] *=0
    W *= np.sqrt(dt)
    W = np.cumsum(W)*dt
    return [np.sum(np.sin(k*times)*W) for k in ks]

def testFourierConvergence():

    L = 10.0
    c = lambda k: 0.5*cCovExp(L,10.0,k)

    inputList = list(itertools.product(range(5),range(5)))
    inputList = [list(foo) for foo in inputList]
    M = 100

    #samples = [fourierExample(inputList,c,L=L) for m in range(M)]
    
    N1=10
    N2=10

    rate1 =  np.array([fourierExample([[5,foo] for foo in range(1,N1)],c) for m in range(100)])
    weakrate1 = np.mean(np.abs(np.diff(rate1,axis=1)),axis=0)
    strongrate1 = np.mean(np.diff(rate1,axis=1)**2,axis=0)
    
    rate2 =  np.array([fourierExample([[foo,5] for foo in range(1,N2)],c) for m in range(100)])
    weakrate2 = np.mean(np.abs(np.diff(rate2,axis=1)),axis=0)
    strongrate2 = np.mean(np.diff(rate2,axis=1)**2,axis=0)
    
    plt.figure()
    plt.semilogy(range(1,N1-1),weakrate1)
    plt.grid(1)
    plt.xlabel('$log_2 N_f$')
    plt.ylabel('$E |g-\overline{g}|$')
    plt.savefig('nfweak.pdf')

    plt.figure()
    plt.semilogy(range(1,N1-1),strongrate1)
    plt.grid(1)
    plt.xlabel('$log_2 N_f$')
    plt.ylabel('$E (g-\overline{g})^2$')
    plt.savefig('nfstrong.pdf')
    
    plt.figure()
    plt.semilogy(range(1,N2-1),weakrate2)
    plt.grid(1)
    plt.xlabel('$log_2 N_t$')
    plt.ylabel('$E |g-\overline{g}|$')
    plt.savefig('ntweak.pdf')

    plt.figure()
    plt.semilogy(range(1,N2-1),strongrate2)
    plt.grid(1)
    plt.xlabel('$log_2 N_t$')
    plt.ylabel('$E (g-\overline{g})^2$')
    plt.savefig('ntstrong.pdf')


