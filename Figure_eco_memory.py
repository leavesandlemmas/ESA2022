# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:35:42 2022

@author: fso
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.integrate import solve_ivp


dens= 2
tmax = 7
t0 = 2.5
ite = 10
slow = 0.2
U = 7

s = np.linspace(0.,0.2,201)
t = np.linspace(0,tmax,100*tmax + 1)


T,S = np.meshgrid(t,s)
Vt = np.ones_like(T)

def A(t):
    As =  1  - 0.5*np.cos(2*np.pi*t)
    Au = expit(ite* (t - 0.5 - t0))  + expit(-ite*(t - t0+0.5))
    return  Au * As

def fn(t,s):
    v = A(t)*(1- s) - U*s

    return slow *v

Vs = fn(T,S)

ncl = A(t)/(A(t) + U)
sol = solve_ivp(fn,[-tmax, tmax], [ ncl[0]], t_eval=t)

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].plot(t, slow * A(t), color = 'green')

ax[1].streamplot(T, S, Vt, Vs, density =dens)
ax[1].plot(t, ncl)
ax[1].plot(t, sol.y[0], color = 'k')
ax[1].set_ylim(s[0],s[-1])
plt.show()
