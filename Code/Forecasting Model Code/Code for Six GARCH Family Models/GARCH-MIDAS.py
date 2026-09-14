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
Created on Sun Mar 21 22:41:38 2021

@author: user
"""

from numpy import array,nan,inf,nanmean,nansum,ceil,zeros,ones,exp,arange,any,log,pi,vstack,\
    all,max,abs,diag,isnan,isinf,percentile
from numpy.linalg import inv,pinv
from pandas import DataFrame
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st#scipy.stats称为统计函数库
from statsmodels.tools.numdiff import approx_hess
import numpy as np
import matplotlib.pyplot as plt
import math#导入math模块，运用模块里面的函数
from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import sys


def GarchMidas0(y,K,period,estSample,logTau=False):#当语句以：结尾时，缩进的语句视为代码块（坚持使用4个空格的缩进）
    #y -= nanmean(y) #相减，然后返回值赋给前一个变量，其中nanmean表示出去nan值以外的所有数的平均值
    mu0 = np.nanmean(y);alpha0 = 0.05;beta0 = 0.9;theta0 = 0.01
    w0 = 5;m0 = 0.1;
    #mu0 = np.nanmean(y);alpha0 = 0.05;beta0 = 0.9;theta0 = 0.01
    #w0 = 5;m0 = 0.1;
    y = y.flatten()#只能适用于numpy对象，即array或mat,普通的list列表不适用。a.flatten()：a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降
    yFull = y[:]#x[:]表示x的所有元素。y是x的浅拷贝，会使用新的内存地址来存放拷贝值
    y = y[:estSample]#取得是y的样本值
    nobs = len(y) #返回字符串，列表，字典，元组等长度
    RV = zeros(nobs)#生成一个nobs*nobs的零矩阵
    for t in range(period,nobs):
        RV[t] = nansum(y[t-period:t]**2)#
    RV[:period] = RV[period]
    params0 = [mu0,alpha0,beta0,theta0,w0,m0]
    bound = [(-inf,inf),(0,1),(0,1),(-inf,inf),(1,inf),(-inf,inf)]
    #bound = [(-inf,inf),(0,1),(0,1),(-inf,inf),(1.001,50),(-inf,inf)]
    options = {'maxiter' :3000,'disp':True}#
    #options = {'maxiter':4500,'disp':True}
    myfun = lambda x:-nansum(fML(x,y,RV,K,period,nobs,logTau)[0])#lambda是匿名函数，其中myfun指的是最小值的目标函数
    estParams = minimize(myfun,params0,method='SLSQP',options=options,bounds=bound).x  #minimize指的是局部最优的解法，

#参数估计结果显示的样子
    hess = approx_hess(estParams,myfun)#hess表示黑塞矩阵
    vcv_H = pinv(hess)#pinv指的是黑塞矩阵的广义逆矩阵
    se_H = abs(diag(vcv_H))**0.5#diag表示矩阵的对角线元素，abs表示返回数值绝对值
    zstat = estParams/se_H #浮点数除法
    p_value = st.t.sf(abs(zstat),nobs-len(estParams))
    p_value[p_value<1e-4]=0
    
    index = ['mu','alpha','beta','theta','w','m']
    columns = ['result','stderr','stat','p_value']
    result = vstack((estParams,se_H,zstat,p_value)).T  #vstack(tup)其中tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组，就是垂直的把数组排列起来
    table = DataFrame(result,index=index,columns=columns)
    print(table)

    AIC = (14 - nansum(fML(estParams, y, RV, K, period, nobs,logTau)[0]) * 2) / len(y)
    BIC = (-2 * nansum(fML(estParams, y, RV, K, period, nobs,logTau)[0]) + 7 * np.log(len(y))) / len(y)
    print('AIC是：%f\n' % (AIC))
    print('HQ是：%f\n' % (BIC))

    nobsBig = len(yFull)
    RVBig = zeros(nobsBig)#结果显示的是1*nobsBig的零矩阵
    for t in range(period,nobsBig):
        RVBig[t] = nansum(yFull[t-period:t]**2)
    RVBig[:period] = RVBig[period]

    Variance,LongRun,ShortRun = fML(estParams,yFull,RVBig,K,period,nobsBig,logTau)[1:]#
    realizedY2 = yFull**2
    forecasterror = Variance - realizedY2
    estSampleMse = nanmean(forecasterror[:estSample]**2)
    print('MSE of one-step variance forecast (period 1 to %d):%f \n'%(estSample,estSampleMse))
    if estSample<nobsBig:
        outSampleMse = nanmean(forecasterror[estSample:] ** 2)
        print('MSE of one-step variance forecast (period %d to %d):%f \n'%(estSample+1,nobsBig,outSampleMse))
    return estParams,Variance,LongRun,ShortRun

def BetaWeights(K,param1):
    eps = 2.2204e-16#设置的默认参数
    seq = arange(K,0,-1)#表示开始，结束，步长。是将列表或字符倒过来
    weights = (1-seq/K+10*eps)**(param1-1)
    weights = weights/nansum(weights)
    return weights

def fML(params,y,RV,K,period,nobs,logTau):
    mu0 = params[0]
    alpha0 = params[1]
    beta0 = params[2]
    theta0 = params[3]
    w0 = params[4]
    m0 = params[5]
    intercept = 1-alpha0-beta0
    if intercept<0 or alpha0<0 or alpha0>1 or beta0<0 or beta0>1:
        logL = array([-1e10]*nobs)
        Variance = array([nan]*nobs)
        ShortRun = array([nan]*nobs)
        LongRun = array([nan,nobs])
        return logL,Variance,ShortRun,LongRun
    theta0 = theta0**2
    m0 = m0**2
    ydeflate = y-mu0
    Resisq = ydeflate**2
    ShortRun = ones(nobs)#生成参数为一的数组
    if logTau:
        tauAvg = exp(m0+theta0*nanmean(RV))
    else:
        tauAvg = m0+theta0*nanmean(RV)
    Variance = tauAvg*ones(nobs)
    nlagBig = period*K
    weights = BetaWeights(nlagBig,w0)
    loopStart = nlagBig
    for t in range(loopStart,nobs):#长期趋势，只受已实现波动的影响
        tau = m0 +theta0*(weights.dot(RV[t-nlagBig:t]))#dot表示权重和RV的点积
        if logTau:
            tau = exp(tau)
        alphaTau = (alpha0)/tau
        ShortRun[t] = intercept+alphaTau*Resisq[t-1]+beta0*ShortRun[t-1]#短期趋势
        Variance[t] = tau*ShortRun[t]

    if any(Variance<0):
        logL = array([-1e10]*nobs)#nobs表示用于计算临界值用到的观测值数目
        Variance = array([nan]*nobs)
        ShortRun = array([nan]*nobs)
        LongRun = array([nan,nobs])
        return logL,Variance,ShortRun,LongRun

    logL = -0.5*(log(2*pi*Variance+1e-3)+Resisq/(Variance))
    logL[:period*K] = 0
    if isnan(logL.sum()) or isinf(logL.sum()):
        logL = array([-1e10]*nobs)
    LongRun = Variance/ShortRun
    return logL,Variance,LongRun,ShortRun
 

if __name__=='__main__':
    import matplotlib.pyplot as plt
    y = pd.read_csv(_data('DT50低频.csv'))
    date = y["date"].values
    y = y["rt"].values

    insample=1883
    estParams0, Variance0, LongRun0, ShortRun0 = GarchMidas0(y, 12, 22, insample, False)

    #plt.plot(Variance0,'b', LongRun0 ,'r-',ShortRun0,'y')#linewidth表示线宽
#plt.show()

yf = y[insample:]
yf = yf ** 2
Variance_f = Variance0[insample:]
plt.subplot(311)
plt.plot(yf, 'b', Variance_f, 'r')
plt.title('nihe')

plt.subplot(312)
plt.plot(Variance0, 'b', LongRun0, 'r')
plt.subplot(313)
plt.plot(Variance0, 'b', ShortRun0, 'y')
plt.show()

import pandas as pd
date=date[insample:]
d=[yf,Variance_f]
df = pd.DataFrame({'v':yf, 'GARCH-MIDAS':Variance_f},index=date)
df.to_csv(_out(r'3.csv'))



