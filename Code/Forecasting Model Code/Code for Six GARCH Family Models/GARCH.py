import os as _os
def _out(name):
    d = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'outputs')
    _os.makedirs(d, exist_ok=True)
    return _os.path.join(d, name)

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:42:53 2019


"""

import warnings
warnings.filterwarnings("ignore") #忽略警告
import numpy as np
import pandas as pd
from numpy.linalg import inv,pinv
from scipy.stats import norm,t
from scipy.optimize import minimize
from numpy import inf,array,nan,nansum
from statsmodels.tools.numdiff import approx_hess

def garch(rt,estSample):
    mu = np.mean(rt)
    alpha = 0.1
    beta = 0.9
    omega = 0.1
    rt = rt.flatten()
    rtFull = rt[:]
    rt = rt[:estSample]
    nobs = len(rt)
    params0 = [mu,omega,alpha,beta]
    f = lambda x:-nansum(fML(x,rt,nobs)[0])
    bound = [(-inf,inf),(-inf,inf),(0,1),(0,1)]
    cons = {'type':'ineq','fun':lambda x:1-x[2]-x[3]}
    estParams = minimize(f,params0,method='SLSQP',bounds=bound,constraints=cons).x
    
    hess = approx_hess(estParams,f)
    #vcv_H = np.linalg.inv(hess)
    vcv_H = pinv(hess)
    se_H = abs(np.diag(vcv_H))**0.5
    zstat = estParams/se_H
    p_value = t.sf(abs(zstat),len(rt)-len(estParams))
    p_value[p_value<1e-4]=0
    
    index = ['mu','omega','alpha','beta']
    columns = ['result','stderr','stat','p_value']
    result = np.vstack((estParams,se_H,zstat,p_value)).T
    table = pd.DataFrame(result,index=index,columns=columns)
    print(table)
    
    AIC = (8-nansum(fML(estParams,rt,nobs)[0])*2)/len(rt)
    BIC = (-2*nansum(fML(estParams,rt,nobs)[0])+4*np.log(len(rt)))/len(rt)
    print('AIC是：%f\n'%(AIC))
    print('HQ是：%f\n'%(BIC))
    
    nobsBig = len(rtFull)
    Variance = fML(estParams,rtFull,nobsBig)[1]
    realizedY2 = rtFull**2
    forecasterror = Variance - realizedY2
    estSampleMse = np.nanmean(forecasterror[:estSample]**2)
    print('MSE of one-step variance forecast (period 1 to %d):%f \n'%(estSample,estSampleMse))
    if estSample<nobsBig:
        outSampleMse = np.nanmean(forecasterror[estSample:] ** 2)
        print('MSE of one-step variance forecast (period %d to %d):%f \n'%(estSample+1,nobsBig,outSampleMse))
    return estParams,Variance

def fML(params,rt,nobs):
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    epsilon = rt-mu
    intercept = omega
    if alpha<0 or alpha>1 or beta<0 or beta>1:
        logL = array([-1e10]*nobs)
        Variance = array([nan]*nobs)
        return logL,Variance
    Variance = np.ones(len(rt))#*(omega/(1-alpha-beta))
    for i in range(1,len(rt)):
        Variance[i] = intercept + alpha*epsilon[i-1]**2 + beta*Variance[i-1] 
    if any(Variance<0):
        logL = np.array([-1e10]*len(rt))
        Variance = np.array([np.nan]*len(rt))
        return logL,Variance
    #L = norm.pdf(y,loc=mu,scale=np.sqrt(v))
    #logL = np.sum(np.log(L))
    logL = -0.5*(np.log(2*np.pi*Variance+1e-3)+(rt-mu)**2/(Variance))
    if np.isnan(logL.sum()) or np.isinf(logL.sum()):
        logL = np.array([-1e10] * nobs)
    return logL,Variance
if __name__=='__main__':
    data = pd.read_csv('DT50低频.csv',encoding="gbk")
    rt = data["rt"].values
    insample = 1883
    params, Variance = garch(rt, insample)#1075是测试样本



import matplotlib.pyplot as plt
yf = rt[insample:]
yf = yf ** 2
Variance_f = Variance[insample:]
plt.subplot(311)
plt.plot(yf, 'b', Variance_f, 'r')
plt.title('nihe')
plt.show()

d = [yf, Variance_f]
df = pd.DataFrame({'v': yf, 'GARCH': Variance_f})
df.to_csv(_out(r'1.csv'))