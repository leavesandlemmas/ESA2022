# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:45:59 2022

@author: fso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


dens=1
s = np.linspace(0.,0.125,128)
t = np.linspace(0,3,128)

T,S = np.meshgrid(t,s)
Vt = np.ones_like(T)

A = 0.5*(1 - np.cos(2*np.pi*T))
U = 10
Vs = A*(1-S) - U*S
fig,ax = plt.subplots(1,3,sharex=True,sharey=True ,figsize=(9,3))


ax[0].streamplot(T,S,Vt, 0.05*Vs, density=dens)

ax[1].streamplot(T,S,Vt,0.5*Vs, density=dens)



ax[2].streamplot(T,S,Vt,10*Vs,density=dens)

ax[0].set_ylim(s[0],s[-1])

# ax[1].set_xlabel('TIME (y)')
# ax[0].set_ylabel('NONSTRUCTURAL CARBOHDYRATES\n(mass fraction)')

# ax[0].set_title("Stable\nOne fixed point")
# ax[1].set_title("Not stable\nNo fixed points")
# ax[2].set_title("Unstable\nOne fixed points")

# ax[2].set_title("Bistable\nthree fixed points")
plt.show()
