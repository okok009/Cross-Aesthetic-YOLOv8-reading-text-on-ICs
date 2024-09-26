import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from scipy.stats import multivariate_normal
from scipy.integrate import quad

# Our 2-dimensional distribution will be over variables X and Y
N = 400
img_HW = 1024
H = 100
W = 400
theta = 30 * math.pi / 180
X = np.linspace(0, img_HW, N)
Y = np.linspace(0, img_HW, N)
X, Y = np.meshgrid(X, Y)
# Mean vector and covariance matrix
mu = np.array([img_HW*0.75, img_HW*0.5])
a = (1/12) * (W**2)
b = (1/12) * (H**2)
Sigma = np.array([[ a * math.cos(theta)*math.cos(theta) + b * math.sin(theta)*math.sin(theta) , (1/2)*(a-b)*math.sin(2*theta)],
                   [(1/2)*(a-b)*math.sin(2*theta),  a * math.sin(theta)*math.sin(theta) + b * math.cos(theta)*math.cos(theta)]])
rv = multivariate_normal(mu, Sigma)

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def marginal_pdf_x(x_vals):
    return np.array([quad(lambda Y: rv.pdf([x_val, Y]), 0, img_HW)[0] for x_val in x_vals])

def marginal_pdf_y(y_vals):
    return np.array([quad(lambda X: rv.pdf([X, y_val]), 0, img_HW)[0] for y_val in y_vals])

x_vals = np.linspace(0, img_HW, 100)
y_vals = np.linspace(0, img_HW, 100)
pdf_x = marginal_pdf_x(x_vals)
pdf_y = marginal_pdf_y(y_vals)

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# plot using subplots
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1])
ax_main = plt.subplot(gs[1, 1])
ax_xpdf = plt.subplot(gs[0, 1])
ax_ypdf = plt.subplot(gs[1, 0])
# X轴边际概率密度函数
ax_xpdf.plot(x_vals, pdf_x, color='orange')
ax_xpdf.set_ylabel('Density')

# Y轴边际概率密度函数
ax_ypdf.plot(pdf_y, y_vals, color='orange')
ax_ypdf.set_xlabel('Density')

ax_main.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)

ax_main.grid(False)
ax_main.set_xlim(1)
ax_main.set_ylim(ax_main.get_ylim()[::-1])
ax_main.xaxis.set_label_position('top')
ax_main.xaxis.tick_top()
ax_main.set_xlabel(r'$x$')
ax_main.set_ylabel(r'$y$')

plt.show()