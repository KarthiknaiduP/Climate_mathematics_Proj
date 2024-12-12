import numpy as np
import matplotlib.pyplot as plt

def zfunc(x, y):
    mydat = np.zeros((100, 100, 10))
    for t in range(0, 10):
        for i in range(0, 100):
            for j in range(0, 100):
                mydat[i, j, t] = (np.sin(t + 1) * (1 / np.pi) * np.sin(x[i]) * np.sin(y[j])
                                  + np.exp(-0.3 * (t + 1)) * (1 / np.pi) * np.sin(8 * x[i]) * np.sin(8 * y[j]))
    return mydat

x = y = np.linspace(0., 2. * np.pi, 100)
mydata = zfunc(x, y)

da = np.zeros((np.size(x) * np.size(y), 10))

for i in range(10):
    da[:, i] = np.reshape(mydata[:, :, i], (10000,))

u, s, vh = np.linalg.svd(da, full_matrices=False)
v = np.transpose(vh)

# Time array for plotting
t = np.array(range(1, 11))

plt.plot(t, v[:, 0], color='k', marker='o', label='PC1: 83% Variance')
plt.plot(t, v[:, 1], color='r', marker='o', label='PC2: 17% Variance')
plt.plot(t, -np.sin(t), color='b', marker='o', label='Original Mode 1 Coefficient: 91% Variance')
plt.plot(t, -np.exp(-0.3 * t), color='m', marker='o', label='Original Mode 2 Coefficient: 9% Variance')

plt.ylim(-1.1, 1.1)
plt.xlabel('Time')
plt.ylabel('PC or Coefficient')
plt.yticks([1, 0.5, 0, -0.5, -1])
plt.title('SVD PCs vs. Accurate Temporal Coefficients')
plt.legend(loc='upper left', fontsize=10)

plt.savefig("svd_PC.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

