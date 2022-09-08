import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

A = 1
U = 5

ncl = A/(A + U)

num = 2000
t = np.linspace(0, 2, num+1)
s = np.linspace(0,0.35, num + 1)
T, S = np.meshgrid(t, s)
Vt = np.ones_like(T)
Vs = A*(1 - S) - U*S

fig, ax = plt.subplots(1,2,sharex=True, sharey=True,figsize=(6,3))


ax[0].streamplot(T, S, Vt, Vs, density=(1,4))
ax[0].axhline(ncl, color = 'k', label = 'Attractor')
ax[0].axhline(ncl, color = 'C1', ls='--', label='Nullcline')

# ax[0].set_xlabel('TIME')
# ax[0].set_ylabel('[NSC] (mass fraction)')


def A(t):
    return 1 - np.cos(2*np.pi*t)
def U(t):
    return 5 - 2*np.cos(2*np.pi*t)
Vs = A(T)*(1 - S) - U(T)*S


ax[1].streamplot(T, S, Vt, Vs, density=(1,4))

ncl = A(t)/(A(t) + U(t))

def func(t,s):
    return A(t)*(1 - s) - U(t)*s


def cross(t,s):
    return func(t,s[0])

# sol = solve_ivp(func,[0,2], [0.3], t_eval =t)
# ax[1].plot(sol.t, sol.y[0], color = 'cyan',label='Simulations')

sol = solve_ivp(func,[-10,2], [ncl[0]], t_eval =t, events = cross)
ax[1].plot(sol.t, sol.y[0], color = 'k',label='Attractor')
ax[1].plot(t, ncl, color = 'C1', ls='--',label='Nullcline')

ax[1].plot(sol.t_events[0][-2:],sol.y_events[0][-2:],'ro',label='Extrema')

ax[1].set_ylim(s[0],s[-1])


ax[0].legend()
plt.show()
