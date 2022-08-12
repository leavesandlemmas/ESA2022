# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:35:20 2022

@author: fso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.integrate import solve_ivp

dens= 2
tmax = 2
t0 = 2.5
ite = 10
slow = 0.2
A = 1
U = 5

s = np.linspace(0.,0.25,201)
t = np.linspace(0,tmax,100*tmax + 1)


T,S = np.meshgrid(t,s)
Vt = np.ones_like(T)



def fn(t, s):
    return At(t)*(1 - s) - U * s

def At(t):
   u = 0.7 - np.cos(2*np.pi*t)
   return np.clip(0.5*u,0,10)

def Ut(t):
   u =  - np.cos(2*np.pi*t)
   return 10*np.exp(u)

def gn(t,s):
    return A*(1-s) - Ut(t)*s


d = np.datetime64('2020-01-01') + (t *365.242).astype('timedelta64[D]')

Vs = fn(T,S)
ncl = At(t)/(At(t) + U)
sol = solve_ivp(fn, [-tmax,tmax], [ncl[0]], t_eval=t)

fig, ax =plt.subplots(2,1,sharey=True)
ax[0].streamplot(T, S, Vt, Vs)
ax[0].plot(t, ncl)
ax[0].plot(t, sol.y[0],color='k')


Vs = gn(T,S)
ncl = A/(A + Ut(t))
sol = solve_ivp(gn, [-tmax,tmax], [ncl[0]], t_eval=t)


ax[1].streamplot(T,S,Vt, Vs)
ax[1].plot(t, ncl)
ax[1].plot(t, sol.y[0],color='k')

ax[1].set_ylim(s[0]-0.01,s[-1])

plt.show()
