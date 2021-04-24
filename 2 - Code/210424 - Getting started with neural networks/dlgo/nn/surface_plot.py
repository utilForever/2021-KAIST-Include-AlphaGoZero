import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(-2, 2, 0.3)
Y = np.arange(-2, 2, 0.3)
X, Y = np.meshgrid(X, Y)
R = Y * np.sin(X) - X * np.cos(Y)
Z = np.sin(R)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_zlim(-1.0, 1.0)
ax.zaxis.set_major_locator(LinearLocator(8))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
