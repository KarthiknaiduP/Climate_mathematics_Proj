import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")

# Define grid
x = y = np.linspace(0., 2. * np.pi, 100)

# Create a 3D data array
mydat = np.zeros((100, 100, 10))

def zfunc(x, y):
    for t in range(0, 10):
        for i in range(0, 100):
            for j in range(0, 100):
                mydat[i, j, t] = (
                    np.sin(t + 1) * (1 / np.pi) * np.sin(x[i]) * np.sin(y[j])
                    + np.exp(-0.3 * (t + 1)) * (1 / np.pi) * np.sin(8. * x[i]) * np.sin(8. * y[j])
                )
    return mydat

mydata = zfunc(x, y)

# Create 2D matrix for SVD
da = np.matrix(np.zeros((np.size(x) * np.size(y), 10)))

def twod_func(x, y, a):
    for i in range(0, 10):
        a[:, i] = np.reshape(mydata[:, :, i], (10000, 1))
    return a

da1 = twod_func(x, y, da)

# Perform SVD
u, s, vh = np.linalg.svd(da1, full_matrices=False) 
v = np.transpose(vh)

# Define EOF functions
def dig():
    y1 = np.transpose(u[:, 0])
    u2 = np.reshape(-y1, (100, 100))
    return u2

def dig2():
    y1 = np.transpose(u[:, 1])
    u2 = np.reshape(-y1, (100, 100))
    return u2

u_mat = dig()
u_mat2 = dig2()

# Define custom colormaps
colors = ["crimson", "red", "tomato", "lightpink", "pink", "peachpuff",
          "yellow", "yellowgreen", "limegreen", "white", "lavenderblush",
          "darkorchid", "indigo", "rebeccapurple", "darkmagenta", 
          "mediumvioletred", "black"]
colors.reverse()
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

colors1 = ["crimson", "red", "tomato", "lightpink", "pink", "peachpuff",
           "yellow", "yellowgreen", "limegreen", "white", "lavenderblush",
           "darkorchid", "indigo", "rebeccapurple", "darkmagenta", 
           "mediumvioletred", "black"]
colors1.reverse()
cmap1 = LinearSegmentedColormap.from_list("custom_cmap1", colors1, N=100)

# Define accurate functions for comparison
def z1(x, y):
    return (1. / np.pi) * np.sin(x) * np.sin(y)

def z2(x, y):
    return (1. / np.pi) * np.sin(8. * x) * np.sin(8. * y)

def fcn1(x, y):
    zeros = np.zeros((100, 100))
    for i in range(0, 100):
        for j in range(0, 100):
            zeros[i, j] = z1(x[i], y[j])
    return zeros

def fcn2(x, y):
    zeros = np.zeros((100, 100))
    for i in range(0, 100):
        for j in range(0, 100):
            zeros[i, j] = z2(x[i], y[j])
    return zeros

# Plot setup
size = np.linspace(-2.5e-2, 2.5e-2, 25)
bounds = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
bounds_d = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])

plt.figure(figsize=(12, 10))

# Plot EOF1
plt.subplot(221)
dig = plt.contourf(x, y, u_mat, size, cmap=cmap)
plt.title('SVD Mode 1: EOF1')
plt.ylabel('y')
plt.xticks([0, 1, 2, 3, 4, 5, 6], size=14)
plt.yticks([0, 1, 2, 3, 4, 5, 6], rotation=90, size=14)
cbar = plt.colorbar(dig, ticks=bounds)
cbar.ax.tick_params(labelsize='large')

# Plot Accurate Mode 1
plt.subplot(223)
lvl = np.linspace(-0.35, 0.35, 35)
contour = plt.contourf(x, y, fcn1(x, y), lvl, cmap=cmap1)
plt.title('Accurate Mode 1')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([0, 1, 2, 3, 4, 5, 6], size=14)
plt.yticks([0, 1, 2, 3, 4, 5, 6], rotation=90, size=14)
cbar = plt.colorbar(contour, ticks=bounds_d)
cbar.ax.tick_params(labelsize='large')

# Plot EOF2
plt.subplot(222)
dig2 = plt.contourf(x, y, np.flip(u_mat2, axis=1), size, cmap=cmap)
plt.title('SVD Mode 2: EOF2')
plt.xticks([0, 1, 2, 3, 4, 5, 6], size=14)
plt.yticks([0, 1, 2, 3, 4, 5, 6], rotation=90, size=14)
cbar = plt.colorbar(dig2, ticks=bounds)
cbar.ax.tick_params(labelsize='large')

# Plot Accurate Mode 2
plt.subplot(224)
contour2 = plt.contourf(x, y, fcn2(x, y), lvl, cmap=cmap1)
plt.title('Accurate Mode 2')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([0, 1, 2, 3, 4, 5, 6], size=14)
plt.yticks([0, 1, 2, 3, 4, 5, 6], rotation=90, size=14)
cbar = plt.colorbar(contour2, ticks=bounds_d)
cbar.ax.tick_params(labelsize='large')

plt.savefig("svd_visualization.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

