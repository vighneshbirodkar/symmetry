import numpy as np
from matplotlib import pyplot as plt

def kernel(width, sigma, theta):

    eps = 4 # 1 peak

    xx, yy = np.meshgrid(np.arange(-width, width + 1), np.arange(-width, width + 1) )
    xx = xx.astype(np.float)
    yy = yy.astype(np.float)
    #print(yy)
    u2 = xx*xx + yy*yy

    ue_theta = xx*np.cos(theta) + yy*np.sin(theta)

    e1 = (1j*ue_theta*2*np.pi/(eps*sigma))
    exp1 = np.exp(e1)
    exp2 = np.exp(-u2/(2*sigma*sigma))

    c2 = np.sum(exp1*exp2)/ np.sum(exp2)

    psi = (1.0)/sigma*(exp1 - c2)*exp2

    c1 = 1.0/np.sqrt(np.sum(psi*np.conj(psi)))

    return c1*psi

def normalize(arr):
    mn, mx = np.min(arr), np.max(arr)
    return (arr - mn)/(mx - mn)
