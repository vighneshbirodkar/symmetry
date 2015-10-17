from skimage import io, color, util, draw
import symmetry
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.util.colormap import viridis
import morlet
import time


name = '/home/vighnesh/images/symmetry/S/I016.png'
img_in = io.imread(name)
img_in = util.img_as_float(img_in)[:,:,0:3]

if img_in.ndim == 2:
    img_in = color.gray2rgb(img_in)

img = color.rgb2gray(img_in)
T = time.time()
sym, d, angle_bins, vote_map = symmetry.symmetry(img, min_dist=2, max_dist=80)
print('Time = ', time.time() - T)

x1,y1,x2,y2 = symmetry.line_coords(img_in, sym, d, angle_bins)
r, t = np.unravel_index(np.argmax(sym), sym.shape)

#
xr = np.arange(0,sym.shape[1])
yr = np.arange(0,sym.shape[0])
xx, yy = np.meshgrid(xr, yr)
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#
ax.plot_surface(xx, yy, sym, cstride=1, rstride=1, linewidth=0, antialiased=False, cmap=cm.coolwarm)

plt.figure()
line = draw.line(y1, x1, y2, x2)
draw.set_color(img_in, line, (1,0,0))
plt.imshow(vote_map, cmap='gray')
plt.figure()
plt.imshow(img_in)
plt.show()
