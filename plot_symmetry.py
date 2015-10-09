from skimage import io, color, util, draw
import symmetry
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.util.colormap import viridis


def line_coords(img, sym, dist, angle_bins):
    img = img[:,:,0:3]

    r, t = np.unravel_index(np.argmax(sym), sym.shape)
    #print("max = ", sym.max(axis=0))
    line_angle = angle_bins[t] - np.pi/2
    offset = sym.shape[0]/2
    ymax, xmax, ch = img.shape

    if line_angle==0:
        x1 = r - offset
        y1 = 0
        x2 = r - offset
        y2 = ymax-1
    elif line_angle == -np.pi/2:
        x1 = 0
        y1 = -(r - offset)
        x2 = xmax - 1
        y2 = -(r-offset)
    else:
        #line_angle = np.pi - line_angle
        m = -np.cos(line_angle)/np.sin(line_angle)
        c = (r - offset)/np.sin(line_angle)
        # y = mx + c
        x1 = -c/m
        y1 = 0
        x2 = (ymax-1 - c)/m
        y2 = ymax-1


    #print('dist = ', r - offset, "angle = ", angle_bins[t]*180/np.pi)
    x1,y2,x2,y2 = map(int, (x1,y1,x2,y2))
    #print("sart",x1,y1)
    #print("end",x2,y2)

    return x1,y1,x2,y2
    #line = draw.line(y1, x1, y2, x2)
    #draw.set_color(img, line, (1,0,0))
    #plt.imshow(img, cmap='gray')
    #io.imsave('/home/vighnesh/Desktop/mail/6.png', img)
    #plt.figure()
    #plt.plot(dist)

    #plt.show()


N = 3

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

    #print('Narrowing search to (%d, %d)' % (i-1,j+1))
    #sym, d, angle_bins = symmetry.symmetry(img, min_dist=i, max_dist=j)

    x1,y1,x2,y2 = line_coords(img_in, sym, d, angle_bins)
    line = draw.line(y1, x1, y2, x2)
    draw.set_color(img_in, line, (1,0,0))

    #io.imsave(('out/I%03d.png' % img_id), img_in)
    #plt.imshow(img_in)
    #plt.imshow(sym)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xr = np.arange(0,sym.shape[1])
    yr = np.arange(0,sym.shape[0])
    xx, yy = np.meshgrid(xr, yr)
    print(xx.shape, yy.shape, sym.shape)
    ax.plot_surface(xx, yy, sym, cstride=1, rstride=1, linewidth=0, antialiased=False, cmap=cm.coolwarm)

    plt.show()

    input()

#display(img_in, sym, d, angle_bins)
