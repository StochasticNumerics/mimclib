#!/usr/bin/python

import numpy as np
import numpy.linalg as npla
import scipy as sp
import copy
import matplotlib.pyplot as plt

class SchoebelZhuHullWhite():

    '''
    A class that describes the stochastic process defined in eq. 2.1
    of http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/Hybrid_SZHW.pdf

    '''

    def __init__(self,X0,p,lam,theta,eta,kappa,sbar,gam,corrMat):
        self.X0 = 1.0*X0
        self.p = 1.0*p
        self.lam = 1.0*lam
        self.kappa = 1.0*kappa
        self.theta = theta
        self.eta = 1.0*eta
        self.sbar = 1.0*sbar
        self.gam = 1.0*gam
        self.corrMat = 1.0*corrMat
        self.G = npla.cholesky(self.corrMat)
    
    def cumulativeVariance(self,dt,X):
        '''
        Return the time series of cumulative variance
        '''
        rv = X[:-1,2]**2*dt
        rv = np.cumsum(rv)
        return np.concatenate(([0.0],rv))

    def timeGrid(self,dt,X):
        '''
        Given a mesh spacing dt and a vector X, the function
        returns the time mesh on which X is defined
        '''

        return dt*np.array(range(len(X)))

    def plotTrajectory(self,dt,X,cumTarg=0.01):
        '''
        Plot the trajectories relative to their initial point
        Colour coding:
        Black - Asset price
        Blue - Interest rate
        Red - Volatility
        Green - Cumulative volatility
        '''

        times = self.timeGrid(dt,X)
        plt.plot(times,X[:,0]/self.X0[0],'k-')
        plt.plot(times,X[:,1]/self.X0[1],'b-')
        plt.plot(times,X[:,2]/self.X0[2],'r-')
        cumVar = self.cumulativeVariance(dt,X)
        plt.plot(times,cumVar/cumTarg,'g-')
        return 0

    def feRealisation(self,dt,targetVar,minT,refine=True):
        '''
        Compute an Euler-Maruyama realisation of the proces with
        a mesh spacing dt
        '''

        Nt = max(int(minT/dt)+1,2)
        rv = np.zeros((Nt,3))
        W = np.zeros((Nt,3))
        rv[0,:] = 1*self.X0
        for ii in range(1,Nt):
            W[ii,:] = W[ii-1,:]
            rv[ii,1] = (rv[ii-1,1]+self.lam*dt*self.theta(dt*ii))/(1+self.lam*dt)
            rv[ii,2] = (rv[ii-1,2]+self.kappa*dt*self.sbar)/(1+self.kappa*dt)
            rv[ii,0] = rv[ii-1,0]/(1-rv[ii,1]*dt)
            dW = np.sqrt(dt)*np.dot(self.G,sp.randn(3))
            W[ii,:] += dW
            rv[ii,0] += (rv[ii-1,2]**self.p)*rv[ii-1,0]*dW[0]
            rv[ii,1] += self.eta*dW[1]
            rv[ii,2] += self.gam*(rv[ii-1,2]**(1-self.p))*dW[2]
        while self.cumulativeVariance(dt,rv)[-1] < targetVar:
            rv = np.concatenate((rv,0*rv[1:,:]))
            W = np.concatenate((W,0*W[1:,:]))
            Nt = len(rv)
            for ii in range(Nt/2+1,Nt):
                W[ii,:] = W[ii-1,:]
                rv[ii,1] = (rv[ii-1,1]+self.lam*dt*self.theta(dt*ii))/(1+self.lam*dt)
                rv[ii,2] = (rv[ii-1,2]+self.kappa*dt*self.sbar)/(1+self.kappa*dt)
                rv[ii,0] = rv[ii-1,0]/(1-rv[ii,1]*dt)
                dW = np.sqrt(dt)*np.dot(self.G,sp.randn(3))
                W[ii,:] += dW
                rv[ii,0] += (rv[ii-1,2]**self.p)*rv[ii-1,0]*dW[0]
                rv[ii,1] += self.eta*dW[1]
                rv[ii,2] += self.gam*(rv[ii-1,2]**(1-self.p))*dW[2]
                #if self.cumulativeVariance(dt,rv)[-1] > targetVar:
                #    break
        
        #print('var1')
        #print(self.cumulativeVariance(dt,rv)[-1])

        #if not abs(rv[-1,-1]):
        #    ci = -1
        #    while not abs(rv[ci,-1]):
        #        ci -=1
        #    rv = rv[:ci+2]
        #    W = W[:len(rv)]

        #print('var1, second time')
        #print(self.cumulativeVariance(dt,rv)[-1])

        if not refine:
            return (rv,W)

        W2 = 1*W[::2]
        rv2 = 1*rv[::2]
        dt2 = dt*2
        Nt2 = len(rv2)

        for ii in range(1,Nt2):
            rv2[ii,1] = (rv2[ii-1,1]+self.lam*dt2*self.theta(dt2*ii))/(1+self.lam*dt2)
            rv2[ii,2] = (rv2[ii-1,2]+self.kappa*dt2*self.sbar)/(1+self.kappa*dt2)
            rv2[ii,0] = rv2[ii-1,0]/(1-rv2[ii,1]*dt2)
            #print(ii,len(W2),Nt2)
            dW2 = W2[ii,:]-W2[ii-1,:]
            rv2[ii,0] += (rv2[ii-1,2]**self.p)*rv2[ii-1,0]*dW2[0]
            rv2[ii,1] += self.eta*dW2[1]
            rv2[ii,2] += self.gam*(rv2[ii-1,2]**(1-self.p))*dW2[2]

        # At this stage, we have the W2 paths and the real paths too, but the variance budget
        # might not be fulfilled
        while self.cumulativeVariance(dt2,rv2)[-1] < targetVar:
            rv2 = np.concatenate((rv2,np.zeros((1,3))))
            W2 = np.concatenate((W2,np.zeros((1,3))))
            rv2[-1,1] = (rv2[-2,1]+self.lam*dt2*self.theta(dt2*ii))/(1+self.lam*dt2)
            rv2[-1,2] = (rv2[-2,2]+self.kappa*dt2*self.sbar)/(1+self.kappa*dt2)
            rv2[-1,0] = rv2[-2,0]/(1-rv2[-1,1]*dt2)
            dW2 = np.sqrt(dt2)*np.dot(self.G,sp.randn(3))
            W2[-1,:] = W2[-2,:]+dW2
            rv2[ii,0] += (rv2[ii-1,2]**self.p)*rv2[ii-1,0]*dW2[0]
            rv2[ii,1] += self.eta*dW2[1]
            rv2[ii,2] += self.gam*(rv2[ii-1,2]**(1-self.p))*dW2[2]
        #print('var2')
        #print(self.cumulativeVariance(dt2,rv2)[-1])
        minCumVar = min(self.cumulativeVariance(dt,rv)[-1],self.cumulativeVariance(dt2,rv2)[-1])
        if minCumVar<targetVar:
            print('ugabuga')
            print(minCumVar)
            return np.nan
        return (rv,W,rv2,W2)

    def discountedPayoff(self,dt,X,targetVar,g):
        '''
        Given an Euler-Maruyama realisation X
        find the time step for which the targetVariance is exceeded for the first time
        evaluate the payoff g on that time step of X and multiply by the appropriate discount
        factor
        '''
        ts = 0
        cumVar = self.cumulativeVariance(dt,X)
        if cumVar[-1] < targetVar:
            return np.nan
        logDiscount = 0.0
        while(cumVar[ts]<targetVar):
            logDiscount += X[ts,1]*dt
            ts += 1
        return g(X[ts])*np.exp(-1*logDiscount)
        
    def feEll(self,ell,g,targetVar):
        '''
        Compute the difference of payoff functions g on different discretisation levels
        '''
        minTime = float(targetVar/(self.X0[-1]**2))
        dt = minTime/(2**ell)
        if not ell:
            realisation = self.feRealisation(dt,targetVar,minTime,refine=False)
            return self.discountedPayoff(dt,realisation[0],targetVar,g)
        realisations = self.feRealisation(dt,targetVar,minTime)
        #print(self.discountedPayoff(dt,realisations[0],targetVar,g),self.discountedPayoff(2*dt,realisations[2],targetVar,g))
        #return realisations
        return self.discountedPayoff(dt,realisations[0],targetVar,g)-self.discountedPayoff(dt*2,realisations[2],targetVar,g)


thetaFun = lambda t: 0.02
X0 = np.array([100.0,0.01,0.2])
K = 1*X0[0]
g = lambda x: max(x[0]-K,0.0)
p = 1.0
gam = 0.0
eta = 0.0
kappa = 0.0
lam = 0.0
sbar = 0.15
corrMat = np.eye(3)
testModel = SchoebelZhuHullWhite(X0, p, lam, thetaFun, eta, kappa, sbar, gam, corrMat)
targetVar = 0.04
dtEll = [targetVar/(testModel.X0[2]**2)/(2**foo) for foo in range(0,20)]
mlRealisation = lambda ell: testModel.feEll(ell,g,targetVar)

testEll = 5
qq = mlRealisation(testEll)

M0 = 5000*(2**(testEll+1))
MEll=[int(M0*(2**(-1*foo))) for foo in range(0,testEll+1)]

means = [0]
variances = [0] 

for ell in range(1,testEll+1):
    print('Generating %d samples on level %d...'%(MEll[ell],ell))
    sample = np.array([mlRealisation(ell) for foo in range(MEll[ell])])
    print('...done.')
    means.append(np.mean(sample))
    variances.append(np.var(sample))
    print('Mean %f and variance %f'%(means[-1],variances[-1]))

mean = np.sum(means)
err = 0.0
for ell in range(1,testEll+1):
    err += np.sqrt(variances[ell]/MEll[ell])
print('Option price %f and error %f'%(mean,err))
