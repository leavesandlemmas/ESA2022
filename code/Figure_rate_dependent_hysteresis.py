# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:38:26 2022

@author: fso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.dates as mdates



def integrate(A,U):
    
    def fn(t,s):
        return A(t)*(1-s) - U(t)*s
    
    def fn_jac(t,s):
        return - A(t) - U(t)
    
    def nullcline(t):
        return A(t)/(A(t) + U(t))
    
    
    t = np.linspace(0,5,101)
    sol = solve_ivp(fn, [-2,5],[nullcline(-2)], dense_output=True, t_eval=t)
    return sol, nullcline(t)

def sample(A, U, sol, num=50, tspan=[0,5]):
    ts = tspan[0] + np.random.rand(num)*(tspan[1] - tspan[0])
    z = np.random.randn(2,2,num)
    err = pow(10,0.025*z[0])*(1 + 0.1*z[1])
    return ts,err 

def to_date(t):
    d = (t * 365.242).astype(np.int64)
    return d.astype('datetime64[D]')

def plot(ax, A, U):
    sol,ncl = integrate(A,U)
    t,s = sol.t,sol.y[0]
    ts, err= sample(A, U, sol)
    ss = sol.sol(ts)[0]
    
    date = to_date(t)
    dates = to_date(ts)
    ax[0].plot(date, s, 'k')
    ax[0].plot(date, ncl, 'k--')
    ax[0].scatter(dates, ss*err[0])
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    ax[1].plot(A(t), s, 'k')
    ax[1].plot(A(t), ncl, 'r')
    ax[1].scatter(A(ts)*err[0], ss*err[1])
    
    ind = np.argsort(ts)
    ax[2].plot(dates[ind], ss[ind]*err[1,ind],'-o')
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax[3].scatter(A(ts)*err[0], ss*err[1])
    
    for ax__ in ax:
        ax__.grid()


fig, ax = plt.subplots(2,4)

def A(t):
    a = 0.1*np.sin(2*np.pi/5*t) + 0.1*np.cos(2*np.pi/10*t)
    u = 2*np.pi*t
    out = 1 - np.cos(u)
    return 0.4*out*np.exp(a)


def U(t):
    a = 0.1*np.cos(2*np.pi/5 * t)
    u = 0.5*np.cos(2*np.pi*(t-0.4))
    return 15*np.exp(a + u)

plot(ax[0],A,lambda t : 10.)

plot(ax[1],lambda t : 0.25*A(t),U)

plt.show()