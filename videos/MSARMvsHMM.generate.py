import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as pp

###############
#  Parmaeters
###############

A0 = [[0.9, 0.0], [0.0, 0.9]]
A1 = [[0.9, 0.0], [0.0, 0.9]]

u0 = [0.0, -2.0]
u1 = [0.0, 2.0]

sigma0 = [[0.1, 0.0], [0.0, 0.1]]
sigma1 = [[0.1, 0.0], [0.0, 0.1]]

T = [[0.99, 0.01], [0.01, 0.99]]

n_steps = 1000

#################
#################

def pplot(x, i):
    points = x[i-plot_lag+1:i+1, :].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cm.bone_r)
    lc.set_array(np.arange(plot_lag))
    lc.set_linewidth(2)
    pp.gca().add_collection(lc)
    # plot the head
    pp.scatter(x[i,0], x[i, 1], c='k', s=30)

    pp.xlim(-2, 2)
    pp.ylim(-4, 4)
    pp.xticks([-2, -1, 0, 1, 2])
    pp.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])


#################
# Script
#################

# stack arrays
A = np.array([A0, A1])
u = np.array([u0, u1])
sigma = np.array([sigma0, sigma1])

# output goes here
x = np.zeros((n_steps, 2), dtype=np.float)  # MSAM
y = np.zeros((n_steps, 2), dtype=np.float)  # "standard" HMM
s = np.zeros(n_steps, dtype=np.int)         # State sequence

fig1 = pp.figure()
plot_lag = 20
#plot = False
plot = True

if plot:
    pp.figure(figsize=(7, 6))


for i in range(1, n_steps):
    x_old = x[i-1]
    s_old = s[i-1]
    
    s_new = np.random.choice([0,1], p=T[s_old])
    x_new = np.dot(A[s_new], x_old - u[s_new]) + np.random.multivariate_normal(u[s_new], sigma[s_new])

    x[i] = x_new
    y[i] = np.random.multivariate_normal(u[s_new], sigma[s_new]/(1-A[0,0,0]**2))
    s[i] = s_new

    if (i > plot_lag) and plot and (i % 2) == 0:
        pp.clf()
        print i
        pp.subplot(1,2,1)
        pplot(x, i)
        pp.subplot(1,2,2)
        pplot(y, i)
        pp.savefig('movie2/movie-%05d.png' % i)
