# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:33:05 2022

@author: fso
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


from scipy.integrate import solve_ivp


s0 =0.8
A = 1.
U = 1.2222
r = A+ U
ncl = A/(A + U)
wait = 50

fig, ax = plt.subplots()

blackbox = patches.Rectangle((0.2,0.05),0.6,0.9,facecolor='white', edgecolor = 'black', lw= 2)

water = patches.Rectangle((0.2,0.05),0.6,s0, facecolor = 'C0')



def fn(t,s):
    return A*(1 - s) - U*s

def fn_jac(t,s):
    return - np.array([[A + U]])


t = np.linspace(0,2.5*np.log(10)/r,201)
sol =solve_ivp(fn, [t[0],t[-1]], [s0],method='BDF', jac=fn_jac, t_eval=t)


def init():
    
    ax.add_patch(blackbox)
    ax.add_patch(water)
    ax.axhline(0.05+ncl, ls='--', color = 'k')
    return water,
    
def animate(i):
    if i % (t.shape[0] + wait) < wait:
        return water,   
    water.set_height(sol.y[0,i-wait])
    return water,
    


ax.axis('off')

ani = FuncAnimation(fig, animate, init_func=init, frames=t.shape[0] + wait, interval= 3, blit=True)
# ani.save('bucketfalling.gif')
plt.show()