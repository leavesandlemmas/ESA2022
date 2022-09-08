'''
Simulation of angiosperm starch + phenology

author: Scott Owald
date: 2022-08-22
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import vectorize

# helper functions
def make_sim(A, U):

    # vector field (diff. eq.); NSC dyn equation
    def sim(t, s):
        return A(t)*(1 - s) - U(t)*s

    # nullcline
    def ncl(t):
        return A(t)/( A(t) + U(t) )

    # response rate
    def rate(t):
        return A(t) + U(t)

    return sim, ncl, rate

def simulate(A, U, t, t0=-1):
    sim, ncl, r = make_sim(A, U)

    # calculate an integral curve
    sol = solve_ivp(sim, [t0,t[-1]], [0.1], method='BDF', t_eval=t)

    return sol.y[0], ncl(t), r(t)

# make A(t) function
def makeA(Amax, t0, t1):
    '''
    Return A(t)

    Amax : summer photosynthesis rate
    t0 : leaf flush out date
    t1 : leaf scenescene date
    '''
    @vectorize('float64(float64)') # this decorator isn't needed for simulation
    def A(t):
        return Amax*(t0 <= (t % 1) < t1)
    return A

# make U(t) function
def makeU(Umax, Umin, t0, t1):
    '''
    Return U(t)

    Umax : summer usage rate
    Umin : winter usage rate
    t0 : growing season start date
    t1 : growing season end date
    '''
    @vectorize('float64(float64)')
    def U(t):
        return Umin + (Umax - Umin)*(t0 <= (t % 1) < t1)
    return U

# time
t = np.linspace(0, 2, 101)

# simulations
A0 = makeA(2, 0.25, 0.75) # 0.25 is early April; 0.75 is early October
U0 = makeU(15, 1, 0.15, 0.75) # 0.15 is late February
y, nl,rr = simulate(A0, U0, t, t0=-3)

# PLOTTING
fig, ax = plt.subplots(4,1, sharex=True)

ax[0].plot(t, A0(t), 'C0', label='A')

ax[1].plot(t, U0(t), 'C3', ls = 'dashed', label='U')
ax[1].plot(t, rr, ls= 'dashdot', color='k',label='Response Rate')

ax[2].plot(t, A0(t)*(1 - y) , label = "Supply A*(1-s)")
ax[2].plot(t, U0(t)*y , color='C3', ls='dashed'  , label = "Demand U*s")

ax[3].plot(t, y, '-k', label = '[starch]')
ax[3].plot(t, nl,'--k',label='nullcline')

for ax__ in ax.ravel():
    ax__.legend()
    ax__.grid()

plt.show()
