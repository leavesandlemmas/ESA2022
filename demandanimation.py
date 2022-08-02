# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:00:20 2022

@author: fso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation



def make(amp):
    def U(t):
        return 5 - amp*np.cos(2*np.pi*t)
    
    def fn(t,s):
        return A*(1 - s) - U(t) *s 
    return U, fn

amps = np.linspace(0,4.5,101)


amp = 0
A = 0.7
ncl_avg = A/(A + 5)
t = np.linspace(0,2,101)
s = np.linspace(0,3*ncl_avg, 101)

T,S  = np.meshgrid(t,s)
Vt = np.ones_like(T)
U, fn = make(0)
Vs = fn(T,S)


sol = solve_ivp(fn, [-1,2], [ncl_avg], method='BDF', t_eval=t)

fig, ax = plt.subplots()
stream = ax.streamplot(T,S,Vt, Vs)


ncl = A/(U(sol.t) + A)
line_ncl, = ax.plot(sol.t, ncl, color = 'red', ls= '--')
line_nsc, = ax.plot(sol.t,sol.y[0], color = 'k')

ax.set_xlabel('Time (y)')
ax.set_ylabel('[NSC]')

def animate(i):
    ax.collections = [] # clear lines streamplot
    ax.patches = [] # clear arrowheads streamplot
    
    U, fn = make(amps[i])
    Vs = fn(T,S)
    sol = solve_ivp(fn, [-1,2], [ncl_avg], method='BDF', t_eval=t)
    
    ncl = A/(A + U(sol.t))
    
    line_ncl.set_data(sol.t, ncl)
    line_nsc.set_data(sol.t,sol.y[0])
    stream = ax.streamplot(T, S, Vt, Vs, color = 'C0')
    return line_ncl, line_nsc, stream

ani = FuncAnimation(fig, animate, frames= amps.shape[0], interval = 1)
plt.show()