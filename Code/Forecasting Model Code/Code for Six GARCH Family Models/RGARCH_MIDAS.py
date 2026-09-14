import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, '..', '..', '..'))

def _data(name):
    """Resolve a data file: first next to this script, then under Data/."""
    local = _os.path.join(_HERE, name)
    if _os.path.exists(local):
        return local
    for sub in ('Data',
                _os.path.join('Data', 'Stock Price and Indicator Data'),
                _os.path.join('Data', 'Stock Price and Indicator Data',
                              'Reference Indicator Data')):
        cand = _os.path.join(_ROOT, sub, name)
        if _os.path.exists(cand):
            return cand
    raise FileNotFoundError(name)

def _out(name):
    d = _os.path.join(_ROOT, 'outputs')
    _os.makedirs(d, exist_ok=True)
    return _os.path.join(d, name)

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:32:23 2019

"""

from numpy import array,nan,inf,nanmean,nansum,ceil,zeros,ones,exp,arange,any,log,pi,vstack,\
    all,max,abs,diag,isnan,isinf
from numpy.linalg import inv,pinv
from pandas import DataFrame
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess
import numpy as np
import warnings
warnings.filterwarnings("ignore") #忽略警告
def RGarchMidas(rt,rk,K,period,estSample):
    mu0 = np.nanmean(rt);beta0 = 0.9;alpha0 = 0.1;
    xi = -0.1
    #psi = 0.01
    psi = 0.1
    tao1 = 0.1
    #tao2 = 0.02
    tao2 = 0.1
    deta_u = 1
    #theta0 = 0.01
    theta0 = 0.1
    omega = 0.1;m0 = 0.1;w1=3
    rtFull = rt[:]
    rt = rt[:estSample]
    rkFull = rk[:]
    rk = rk[:estSample]
    nobs = len(rt)
    RV = zeros(nobs)
    for t in range(period,nobs):
        RV[t] = nansum(rt[t-period:t]**2)
    RV[:period] = RV[period]
    params0 = [mu0,omega,alpha0,beta0,xi,psi,tao1,tao2,m0,theta0,deta_u,w1]
    bound = [(-inf,inf),(-inf,inf),(0,0.2),(0.7,1),(-inf,inf),(0,1),(-inf,0),(0,1),(0,2),(-inf,inf),(0,inf),(1.001,50)]
    options = {'maxiter':3000,'disp':True}
    cons = {'type':'ineq','fun':lambda x:1-x[2]-x[3]}
    myfun = lambda x:-nansum(fML(x,rt,rk,RV,K,period,nobs)[0])
    estParams = minimize(myfun,params0,method='SLSQP',options=options,bounds=bound,constraints=cons).x
    """检验"""
    hess = approx_hess(estParams,myfun)
    vcv_H = pinv(hess)
    se_H = abs(diag(vcv_H))**0.5
    zstat = estParams/se_H
    p_value = st.t.sf(abs(zstat),nobs-len(estParams))
    p_value[p_value<1e-4]=0
    
    index = ['mu','omega','alpha','beta','xi','psi','tao1','tao2','m','theta','deta_u','w1']
    columns = ['result','stderr','stat','p_value']
    result = vstack((estParams,se_H,zstat,p_value)).T
    table = DataFrame(result,index=index,columns=columns)
    print(table)
	
    AIC = (24-nansum(fML(estParams,rt,rk,RV,K,period,nobs)[0])*2)/len(rt)
    BIC = (-2*nansum(fML(estParams,rt,rk,RV,K,period,nobs)[0])+12*np.log(len(rt)))/len(rt)
    print('AIC是：%f\n'%(AIC))
    print('HQ是：%f\n'%(BIC))
    nobsBig = len(rtFull)
    RVBig = zeros(nobsBig)
    for t in range(period,nobsBig):
        RVBig[t] = nansum(rtFull[t-period:t]**2)
    RVBig[:period] = RVBig[period]
    Variance,LongRun,ShortRun = fML(estParams,rtFull,rkFull,RVBig,K,period,nobsBig)[1:]
    realizedrt2 = rtFull**2
    forecasterror = Variance - realizedrt2
    estSampleMse = nanmean(forecasterror[:estSample]**2)
    print('MSE of one-step variance forecast (period 1 to %d):%f \n'%(estSample,estSampleMse))
    if estSample<nobsBig:
        outSampleMse = nanmean(forecasterror[estSample:] ** 2)
        print('MSE of one-step variance forecast (period %d to %d):%f \n'%(estSample+1,nobsBig,outSampleMse))
    return estParams,Variance,LongRun,ShortRun

	
def BetaWeights(K,param1):
    #eps = 2.2204e-16#最小的数
    eps =1e-16
    seq = arange(K,0,-1)
    weights = (1-seq/K+10*eps)**(param1-1)
    weights = weights/nansum(weights)
    return weights

def fML(params,rt,rk,RV,K,period,nobs):
    mu0 = params[0]
    omega = params[1]
    alpha0 = params[2]
    beta0 = params[3]
    xi = params[4]
    psi = params[5]
    tao1 = params[6]
    tao2 = params[7]
    m0 = params[8]
    theta0 = params[9]
    deta_u = params[10]
    w0 = params[11]
    theta0 = theta0**2
    m0 = m0**2
    intercept = omega
    if alpha0<0 or alpha0>1 or beta0<0 or beta0>1:
        logL = array([-1e10]*nobs)
        Variance = array([nan]*nobs)
        ShortRun = array([nan]*nobs)
        LongRun = array([nan,nobs])
        return logL,Variance,ShortRun,LongRun
    ShortRun = ones(nobs)
    tauAvg = exp(m0+theta0*nanmean(RV))
    Variance = tauAvg*ones(nobs)
    zt = np.ones(nobs)
    ut = np.ones(nobs)
    nlagBig = period*K
    weights = BetaWeights(nlagBig,w0)
    loopStart = nlagBig
    for t in range(loopStart,nobs):
        tau = m0 +theta0*(weights.dot(RV[t-nlagBig:t]))
        ShortRun[t] = intercept+alpha0*rk[t-1]+beta0*ShortRun[t-1]
        Variance[t] = tau*ShortRun[t]
        zt[t] = (rt[t]-mu0)/np.sqrt(Variance[t])        
        ut[t] = rk[t]-xi-psi*Variance[t]- tao1*zt[t] - tao2*(zt[t]**2) + tao2
        
    if any(Variance<0):
        logL = array([-1e10]*nobs)
        Variance = array([nan]*nobs)
        ShortRun = array([nan]*nobs)
        LongRun = array([nan,nobs])
        return logL,Variance,ShortRun,LongRun
    """
    L1 = norm.pdf(zt,loc=0,scale=1)
    L2 = norm.pdf(ut,loc=0,scale=deta_u)
    logL = np.log(L1*L2)
    logL[:period*K] = 0
    logL=np.sum(logL)
    """
    logL = -0.5*(log(2*pi*Variance+1e-3)+zt**2)-0.5*(log(2*pi*deta_u**2+1e-3)+ut**2/deta_u**2)
    logL[:period*K] = 0
    if isnan(logL.sum()) or isinf(logL.sum()):
        logL = array([-1e10]*nobs)
    LongRun = Variance/ShortRun
    return logL,Variance,ShortRun,LongRun

if __name__=='__main__':
    data = pd.read_csv(_data('DT50低频.csv'))
    rt = data["rt"].values
    rk = data["rk"].values
    insample = 1883
    estParams,Variance, LongRun, ShortRun = RGarchMidas(rt,rk,12, 22,insample)

    import matplotlib.pyplot as plt
    yf = rt[insample:]
    yf = yf ** 2
    Variance_f = Variance[insample:]
    plt.subplot(311)
    plt.plot(yf, 'b', Variance_f, 'r')
    plt.title('nihe')

    plt.subplot(312)
    plt.plot(Variance, 'b', LongRun, 'r')
    plt.subplot(313)
    plt.plot(Variance, 'b', ShortRun, 'y')
    plt.show()

    d = [yf, Variance_f]
    df = pd.DataFrame({'v': yf, 'RGARCH-MIDAS': Variance_f})
    df.to_csv(_out(r'4.csv'))