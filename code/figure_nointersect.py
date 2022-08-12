# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:19:11 2022

@author: fso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.integrate import solve_ivp

dens= 2
tmax = 3
t0 = 2.5
ite = 10
slow = 0.2
U = 7

s = np.linspace(0.,1,201)
t = np.linspace(0,tmax,100*tmax + 1)


T,S = np.meshgrid(t,s)
Vt = np.ones_like(T)

def A(t):
    As =  1  - 0.5*np.cos(2*np.pi*t)
    return  As

def fn(t,s):
    v = A(t)*(1- s) - U*s

    return v


def gn(t, s):
    v = A(t) - 1.5*s/(0.1 + s)
    return v


fig, ax = plt.subplots(1,2, sharey=True)



Vs = gn(T, S)
ncl = A(t) *0.1/(1.5 - A(t))
sol = solve_ivp(gn,[0, tmax], [0.025], t_eval=t)


ax[0].streamplot(T, S, Vt, Vs, density =dens)
ax[0].plot(t, ncl)
ax[0].plot(t, sol.y[0], color = 'k')




Vs = fn(T,S)

ncl = A(t)/(A(t) + U)
sol = solve_ivp(fn,[-tmax, tmax], [ ncl[0]], t_eval=t)

ax[1].streamplot(T, S, Vt, Vs, density =dens)
ax[1].plot(t, ncl)
ax[1].plot(t, sol.y[0], color = 'k')
ax[1].set_ylim(s[0],s[-1])
plt.show()
