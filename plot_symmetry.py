from skimage import io, color, util, draw
import symmetry
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.util.colormap import viridis


for img_id in range(16,17):
    filename = 'S/I%03d.png' % img_id
    print('Processing ' + filename)

    #img_in = io.imread(filename)
    img_in = io.imread('S/I061.png')
    img_in = util.img_as_float(img_in)

    if img_in.ndim == 2:
        img_in = color.gray2rgb(img_in)

    img = color.rgb2gray(img_in)

    sym, d, angle_bins = symmetry.symmetry(img, min_dist=5, max_dist=80)
    print('Initial Guess complete')

    plt.plot(d)


    d_prob = d/np.sum(d)
    max_idx = np.argmax(d_prob)
    max_prob = d_prob.max()

    j = max_idx + 1
    while j < d.shape[0] and d_prob[j] > 0.5*max_prob:
        j += 1

    i = max_idx - 1
    while i >= 0 and d_prob[i] > 0.5*max_prob:
        i -= 1


    x1,y1,x2,y2 = symmetry.line_coords(img_in, sym, d, angle_bins)
    line = draw.line(y1, x1, y2, x2)
    draw.set_color(img_in, line, (1,0,0))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xr = np.arange(0,sym.shape[1])
    yr = np.arange(0,sym.shape[0])
    xx, yy = np.meshgrid(xr, yr)
    print(xx.shape, yy.shape, sym.shape)
    ax.plot_surface(xx, yy, sym, cstride=1, rstride=1, linewidth=0, antialiased=False, cmap=cm.coolwarm)

    plt.show()

    input()
