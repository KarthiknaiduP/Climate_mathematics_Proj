import numpy as np 
import matplotlib
from matplotlib import cm as cm1
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os

from matplotlib.animation import FuncAnimation

from scipy import optimize as opt
import math as m
import cartopy 
import netCDF4 as nc
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.patches import Polygon

from datetime import datetime

from scipy.ndimage import gaussian_filter
from sklearn import datasets, linear_model
from sklearn.neighbors import KernelDensity as kd
from sklearn.linear_model import LinearRegression
from pylab import *
warnings.filterwarnings("ignore")


# create two arrays, x and y, ranging from 0 to 2*pi 
# create a 3D array full of zeros for use later

x=y= np.linspace(0,2*np.pi,100)
mydat = np.zeros((100,100,10))

def zfunc(x,y):
    for t in range(0,10):
        for i in range(0,100):
            for j in range(0,100):
                mydat[i,j,t] = (np.sin(t+1)*(1/np.pi)*np.sin(x[i])*np.sin(y[j]) 
                                + np.exp(-0.3*(t+1))*(1/np.pi)*np.sin(8*x[i])*np.sin(8*y[j]))
    
    return mydat
mydata = zfunc(x,y)

maxi = np.linspace(-0.5,0.5,100)

# define the contour plot using previous x, y, and mydata arrays
fig, ax = plt.subplots()
cmap = cm.get_cmap('hsv', 20)
CS = plt.contourf(x,y,mydata[:,:,0],maxi,cmap=cmap)

ax.set_title('Field Change', pad=10)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Add color bar
cbar = plt.colorbar(CS, ax=ax, ticks=[0.4, 0.2, 0, -0.2, -0.4])
plt.text(6.55, 6.45, "Scale", size=20)

# Define update function for animation
def update(t):
    ax.clear()  # Clear the current axes
    CS = ax.contourf(x, y, mydata[:, :, t], maxi, cmap=cmap)  # Update the contour plot for the current time step
    ax.set_title(f'Field at t = {t + 1}', pad=10)  # Update the title with the current time step
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# Create the animation
ani = FuncAnimation(fig, update, frames=range(10), interval=1000, repeat=False)

ani.save('field_synthetic_data_10sec.gif', writer='ffmpeg', fps=1)

# Show the animation
plt.show()




