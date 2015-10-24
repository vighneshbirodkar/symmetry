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
not_found = []

DEBUG = True
mat = loadmat('/home/vighnesh/images/symmetry/reflection_testing/reflection_testing/single/_data.mat')
data = mat['data'].astype(np.int)

for idx in range(1,40):

    name = '/home/vighnesh/images/symmetry/reflection_testing/reflection_testing/single/I_%03d.png' % idx
    mat_name = '/home/vighnesh/images/symmetry/S/I%03d.mat' % idx
    #mat = loadmat(mat_name)
    print('Processing : ' + name)

    true_x1, true_y1, true_x2, true_y2 = data[idx-1]
    #true_x1, true_y1 = mat['segments'][0][0][0].astype(np.int)
    #true_x2, true_y2 = mat['segments'][0][0][1].astype(np.int)
    true_line = utils.Line(true_x1, true_y1, true_x2, true_y2)

    img_in = io.imread(name)
    img_in = util.img_as_float(img_in)

    if img_in.ndim == 2:
        img_in = color.gray2rgb(img_in)

    img_in = img_in[:, :, 0:3]
    img = color.rgb2gray(img_in)

    mreal, mimag = symmetry.compute_morlet(img, num_angles=32, sigma=2.0)
    w = max(img.shape)
    sym, angle_bins = symmetry.symmetry(img, min_dist=1, max_dist=100,
                                           num_angles=32,
                                           morlet_real=mreal,
                                           morlet_imag=mimag)


    lines = utils.line_coords(img_in, sym, angle_bins, num_lines=N, drange = 20)
    # for l in lines:
    #
    #     x,y = symmetry.comput_center(img,min_dist=0, max_dist=80,num_angles=16,
    #                            morlet_real=mreal, morlet_imag=mimag,
    #                            r = l.r, angle=l.theta)
    #     l.cx = x
    #     l.cy = y


    tp = 0
    fp = 0
    fn = 1
    for i in range(N):
        #print(str(i + 1) + 'Line')
        subset = lines[0:i+1]

        true_line.draw(img_in, white)
        if DEBUG:
            lines[i].draw(img_in, color_list[i])
        fn = 1
        tp = 0
        fp = 0

        #print('----------------Lines = ' + str(i + 1))

        found = False
        true_lines = [utils.Line(true_x1, true_y1, true_x2, true_y2)]

        for cur_line in subset:
            k = 0
            while k < len(true_lines):
                dist = true_lines[k].dist_to_inf_line(cur_line)
                angle_diff_deg = true_lines[k].angle_diff_inf_line_deg(cur_line)
                #print('Thresh = ', 0.2*true_lines[k].len)
                #print('Dist = ', dist)
                #print(dist, angle_diff_deg, dist < 0.2*true_lines[k].len, angle_diff_deg < 10)
                if dist < 0.2*true_lines[k].len and angle_diff_deg < 10:
                    tp += 1
                    fn -= 1
                    #raise IndexError
                    if tp > 1:
                        raise ValueError
                    found = True
                    true_lines.remove(true_lines[k])
                else:
                    k += 1

            fp = i + 1 - tp


        #print(tp,fp,fn)
            #print("Dist = ",dist,"angle = ", angle*180/np.pi)

        #print(tp,fp,fn)
        assert tp <= 1
        assert fp >= 0
        assert fn >= 0
        TP[i] += tp
        FP[i] += fp
        FN[i] += fn
        if (i + 1) == N and tp < 1:
            not_found.append(idx)

    if DEBUG:
        fname = '/home/vighnesh/images/symmetry/out/I%03d.png' % idx
        io.imsave(fname, img_in)

    #plt.imshow(img_in)
    #plt.show()


print('TP = ',TP)
print('FP = ',FP)
print('FN = ',FN)
print('not found = ', not_found)

print(TP/(TP + FP))
print(TP/(TP + FN))

plt.plot(TP/(TP + FN), TP/(TP + FP), marker='o')
plt.axes().set_xlim(0,1.2)
plt.axes().set_ylim(0,1.2)

plt.grid(True)

plt.show()
