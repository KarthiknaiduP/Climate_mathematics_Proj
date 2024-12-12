import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Define time array
t = np.array(range(1, 11))

# Define sine function
def sinarr(t):
    rslt = np.zeros(t)
    for ii in range(1, t + 1):
        rslt[ii - 1] = -np.sin(ii)
    return rslt

# Define exponential function
def exparr(t):
    rslt = np.zeros(t)
    for jj in range(1, t + 1):
        rslt[jj - 1] = -np.exp(-0.3 * jj)
    return rslt

# Generate sine and exponential arrays
sin = sinarr(10)
exp = exparr(10)
print('Dot Product:', np.dot(np.transpose(sin), exp))

# Calculate variances and compare them
v1 = np.var(sin)
v2 = np.var(exp)
print("Variance:", v1 / (v1 + v2))

# Perform SVD on a sample dataset (reusing SVD implementation from the earlier example)
x = y = np.linspace(0., 2. * np.pi, 100)
mydata = np.zeros((100, 100, 10))
for t_idx in range(10):
    for i in range(100):
        for j in range(100):
            mydata[i, j, t_idx] = (np.sin(t_idx + 1) * (1 / np.pi) * np.sin(x[i]) * np.sin(y[j])
                                   + np.exp(-0.3 * (t_idx + 1)) * (1 / np.pi) * np.sin(8 * x[i]) * np.sin(8 * y[j]))

da = np.zeros((10000, 10))
for i in range(10):
    da[:, i] = np.reshape(mydata[:, :, i], (10000,))

u, s, vh = np.linalg.svd(da, full_matrices=False)
v = np.transpose(vh)

# Find dot product of PCs
v_1 = np.transpose(v[:, 0])
v_2 = v[:, 1]
print("Dot Product of PCs:", np.dot(v_1, v_2))

# Prepare arrays for matrix multiplication
sdiag = np.diag(s)
vtran = np.transpose(v[:, 0:1])

# Multiply matrices to generate B and B1
B = np.transpose(np.matmul(np.matmul(u[:, 0:1], sdiag[0:1, 0:1]), vtran))
B1 = np.matmul(np.matmul(u, sdiag), np.transpose(v))

# Reshape B for plotting
BB = np.reshape(B[4, :], (100, 100))
scale = np.linspace(-0.4, 0.4, 25)
bounds = [0.4, 0.2, 0, -0.2, -0.4]

cmap = cm.get_cmap('hsv', 15)

# Plot reconstructed fields
fig = plt.figure(figsize=(14, 6))

# Plot 2-mode SVD reconstructed field
plt.subplot(121)
BBplot = plt.contourf(x, y, BB, scale, cmap=cmap)
plt.title('(a) 2-mode SVD reconstructed field t = 5', size=18)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([0, 1, 2, 3, 4, 5, 6])
cbar = plt.colorbar(BBplot, ticks=bounds)
cbar.ax.tick_params(labelsize='large')

# Plot all-mode SVD reconstructed field
new_B = B1.T
BB = np.reshape(new_B[4, :], (100, 100))
plt.subplot(122)
BBplot = plt.contourf(x, y, BB, scale, cmap=cmap)
plt.title('(b) All-mode SVD reconstructed field t = 5', size=18)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([0, 1, 2, 3, 4, 5, 6])
cbar = plt.colorbar(BBplot, ticks=bounds)
cbar.ax.tick_params(labelsize='large')

plt.savefig("svd_reconstructed.png", dpi=300, bbox_inches='tight')


plt.tight_layout()
plt.show()

