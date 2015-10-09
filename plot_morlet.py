from matplotlib import pyplot as plt
import morlet
import numpy as np
from numpy import pi


sigma_list = [1, 3, 6]
theta_list = [0, pi/6, pi/4, pi/3, pi/2, 3*pi/3, 3*pi/4, 5*pi/6]

WIDTH = 16
count = 1
for sigma in sigma_list:
    for theta in theta_list:

        plt.subplot(len(sigma_list), len(theta_list), count)
        kernel = np.imag(morlet.kernel(WIDTH, sigma, theta))
        result = kernel
        plt.imshow(result, cmap='gray')

        count += 1
plt.show()
