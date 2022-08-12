import numpy as np
import matplotlib.pyplot as plt


A = 1
U = 5

ncl = A/(A + U)

num = 2000
t = np.linspace(0,3,num +1)
s = np.linspace(0,0.35,num + 1)
T,S = np.meshgrid(t,s)
Vt = np.ones_like(T)
Vs = A*(1 - S) - U*S



plt.streamplot(T, S, Vt, Vs, density=(1,4))

plt.axhline(ncl, color = 'k')
plt.axhline(ncl, color = 'C1', ls='--')

plt.xlabel('TIME')
plt.ylabel('[NSC] (mass fraction)')

plt.show()
