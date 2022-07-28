import numpy as np
import matplotlib.pyplot as plt

s = np.linspace(0,1,101)

fig, ax =plt.subplots(figsize=(4,4))

ax.plot(s, 1-s, color = "#1f77b4ff")
ax.plot(s, s/(0.1 + s), color = "#ff0000ff")
ax.grid()
plt.show()
