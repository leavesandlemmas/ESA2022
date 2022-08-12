import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def GK(x,y,J,K):
    '''
    Goldbeter-Koshland Kinetics Function
    '''
    yJ = y*J
    xK = x*K
    b = y - x + yJ + xK
    denom = b + np.sqrt(b**2 - 4*(y-x)*xK)
    return 2*x*K/denom

def A(t):
    return 0.5*(1 - 0.9*np.cos(2*np.pi*t))

def fn(t, s):
    return A(t)*(1 - s) - GK(s, 0.1, 0.01, 0.01)

tmax=  2
t = np.linspace(0,tmax,100*tmax + 1)

s = np.linspace(0,0.2,101)

T,S = np.meshgrid(t,s)
Vt = np.ones_like(T)
Vs = fn(T, S)
# supply and demand
# plt.plot(s,  1 - s)
# plt.plot(s, GK(s, 0.1, 0.01,0.01), color = 'C3')
# plt.grid()

sol = solve_ivp(fn,[-tmax, tmax], [0.1], t_eval=t)

plt.streamplot(T, S, Vt, Vs,density=2)
plt.plot(t,sol.y[0],'k')
plt.ylim(s[0],s[-1])
plt.show()
