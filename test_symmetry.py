from skimage import io, color, util, draw
from skimage.color.rgb_colors import *
import symmetry
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.util.colormap import viridis
import morlet
import time
import utils
from scipy.io import loadmat

color_list = [red, blue, green, yellow, aqua, orange, pink, purple, coral, indigo ]

fp = 0
tp = 0
fn = 0
N = 10

FP = np.zeros(N, dtype=np.float)
TP = np.zeros(N, dtype=np.float)
FN = np.zeros(N, dtype=np.float)

DEBUG = True

for idx in range(1,2):

    name = '/home/vighnesh/images/symmetry/S/I%03d.png' % idx
    mat_name = '/home/vighnesh/images/symmetry/S/I%03d.mat' % idx
    mat = loadmat(mat_name)
    print('Processing : ' + name)

    true_x1, true_y1 = mat['segments'][0][0][0].astype(np.int)
    true_x2, true_y2 = mat['segments'][0][0][1].astype(np.int)
    true_len = np.hypot(true_y1 - true_y2, true_x1 - true_x2)
    true_angle = np.arctan2(true_y1 - true_y2, true_x1 - true_x2)
    true_angle = abs(true_angle%np.pi)
    true_angle_deg = true_angle*180/np.pi
    #print("True Angle = ", true_angle*180/np.pi)

    true_cx, true_cy = (true_x1 + true_x2)/2.0, (true_y1 + true_y2)/2.0
    img_in = io.imread(name)
    img_in = util.img_as_float(img_in)

    if img_in.ndim == 2:
        img_in = color.gray2rgb(img_in)

    img_in = img_in[:, :, 0:3]

    img = color.rgb2gray(img_in)
    sym, d, angle_bins = symmetry.symmetry(img, min_dist=2, max_dist=80)


    lines = utils.line_coords(img_in, sym, d, angle_bins, num_lines=N, drange = 0.2*true_len)

    tp = 0
    fp = 0
    fn = 1
    for i in range(N):
        #print(str(i + 1) + 'Line')
        subset = lines[0:i+1]

        if DEBUG:
            x1,y1,x2,y2,r,t = lines[i]
            line = draw.line(y1,x1,y2,x2)
            draw.set_color(img_in, line, color_list[i])



        fn = 1
        tp = 0
        fp = 0

        #print('----------------Lines = ' + str(i + 1))
        for j in range(len(subset)):
            x1,y1,x2,y2,r,t = subset[j]

            dist = utils.dist_point_line(true_cx, true_cy, x1,y1,x2,y2)
            #print('Dist = ' + str(dist))

            angle = angle_bins[t]
            angle_deg = angle*180/np.pi

            # in single line case, prevent 2 lines from being detected
            if tp < 1 and utils.angle_diff(angle_deg, true_angle_deg) < 10 and abs(dist) < 0.2*true_len:
                tp += 1
                fn -= 1
                if fp > 1:
                    raise ValueError('dist = ' + str(dist) + ", thresh = ", 0.2*true_len)
                #print('True')
            else:
                fp += 1


        #print(tp,fp,fn)
            #print("Dist = ",dist,"angle = ", angle*180/np.pi)

        #print(tp,fp,fn)
        TP[i] += tp
        FP[i] += fp
        FN[i] += fn

    if DEBUG:
        fname = '/home/vighnesh/images/symmetry/out/I%03d.png' % idx
        io.imsave(fname, img_in)


    #plt.imshow(img_in)
    #plt.show()


print('TP = ',TP)
print('FP = ',FP)
print('FN = ',FN)

print(TP/(TP + FP))
print(TP/(TP + FN))
plt.plot(TP/(TP + FN), TP/(TP + FP), marker='o')
plt.axes().set_xlim(0,1.2)
plt.axes().set_ylim(0,1.2)

plt.show()
