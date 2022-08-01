# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:51:38 2022

@author: fso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


dens=1
s = np.linspace(0.,0.35,128)
t = np.linspace(0,3,128)

T,S = np.meshgrid(t,s)
Vt = np.ones_like(T)

def fn(t,s):
    A = 0.5*(1 - np.cos(2*np.pi*t))
    return A*(1-s) - 0.5*0.85/0.15*s

sol = solve_ivp(fn,[0,3],[0.05,0.3],method='BDF',t_eval=t)

fig,ax = plt.subplots(1,3,sharex=True,sharey=True ,figsize=(9,3))

A = 0.5*(1. - np.cos(2*np.pi*T))
U = 0.85/0.15 * 0.5
R = (0.52 + np.cos(2*np.pi*(T-0.6)))

ax[0].streamplot(T,S,Vt, A*(1-S) - U*S,density=dens)
ax[0].plot(0,0.05,'oC1')
ax[0].plot(0,0.3,'oC3')
ax[0].plot(t,sol.y[0],'C1')
ax[0].plot(t,sol.y[1],'C3')


ax[1].streamplot(T,S,Vt, A-R,density=dens)
N = A[0] - R[0]
NI = np.empty_like(N)
NI[0]= 0
NI[1:] = np.cumsum(0.5*(N[1:] + N[:-1])*np.diff(t))

ax[1].plot(0,0.05,'oC1')
ax[1].plot(0,0.3,'oC3')

ax[1].plot(t,0.05 + NI,'C1')
ax[1].plot(t,0.3 + NI,'C3')
ax[1].set_ylim(0,0.35)


# Vs = A*(1 - S) - 1000*S*(0.+(S-0.2)**2)


# def gn(t,s):
#     A = 0.5*(1 - np.cos(2*np.pi*t))
#     return A*(1-s) - 1000*s*(0.0+(s-0.2)**2)



# sol = solve_ivp(gn,[0,3],[0.05,0.3],t_eval=t)

def hn(t,s):
    return -fn(t,s)*0.5

sol = solve_ivp(hn,[0,3],[0.12,0.16],t_eval=t)

Vs = hn(T,S)

ax[2].streamplot(T,S,Vt,Vs,density=dens)

ax[2].plot(0,0.12,'oC1')
ax[2].plot(0,0.16,'oC3')
ax[2].plot(t,sol.y[0],'C1')
ax[2].plot(t,sol.y[1],'C3')

# ax[1].set_xlabel('TIME (y)')
# ax[0].set_ylabel('NONSTRUCTURAL CARBOHDYRATES\n(mass fraction)')

# ax[0].set_title("Stable\nOne fixed point")
# ax[1].set_title("Not stable\nNo fixed points")
# ax[2].set_title("Unstable\nOne fixed points")

# ax[2].set_title("Bistable\nthree fixed points")
plt.show()
