import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

s = np.linspace(0,1,101)

fig, ax =plt.subplots(figsize=(4,4))

A0 = 1.0
V0 = 1.0
Km = 0.1

A1 = 0.5
V1 = 0.9
Km1 = 0.05 

t = np.linspace(0,1,100)

S0 = A0*(1 - s)
S1 = A1*(1 - s)      

D0 = V0*s/(Km + s)
D1 = V1*s/(Km1 + s)

def animate(i):
    u = 2*t[i]**2 if t[i] <= 0.5 else 4*t[i] - 2*t[i]**2 - 1
    S = S0*(1 - u)  + u*S1
    line0.set_data(s,S)
    D =  D0*(1-u) + u*D1
    line1.set_data(s, D)
    return line0, line1
    


line0 = ax.plot(s, A0*(1-s), color = "#1f77b4ff")[0]
line1 = ax.plot(s, V0*s/(Km + s), color = "#ff0000ff")[0]

ax.grid()
ax.set_ylabel('Rate')

ax.set_xlabel('[NSC]')


ani = FuncAnimation(
    fig, animate, frames=t.shape[0], interval=0.1)
ani.save('spldem.gif')
plt.show()