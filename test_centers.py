from skimage import io, color, util, draw
from skimage.color.rgb_colors import *
import symmetry
from matplotlib import pyplot as plt
import numpy as np
from skimage.util.colormap import viridis
import morlet
import utils
from skimage import util
from scipy.io import loadmat

color_list = [red, blue, green, yellow, aqua, orange, pink, purple, coral, indigo ] + [black]*20
BASE_PATH = '/home/vighnesh/images/symmetry/reflection_testing/reflection_testing/multiple/'
DATA_PATH = '/home/vighnesh/images/marcelo2/'
mat = loadmat(BASE_PATH + '_data.mat')
data = mat['data']
PAD = 20
N = 25

FP = np.zeros(N, dtype=np.float)
TP = np.zeros(N, dtype=np.float)
FN = np.zeros(N, dtype=np.float)
pre = np.zeros(N, dtype=np.float)
rec = np.zeros(N, dtype=np.float)
not_found = []

for idx in range(1, 31):
    print(idx)
    img_in = io.imread(BASE_PATH + 'I_%03d.png' % idx)
    img_in = util.img_as_float(img_in)
    img_in = util.pad(img_in, [(PAD, PAD), (PAD, PAD),(0,0)], mode='constant')
    angles = loadmat(DATA_PATH + 'angle_%03d.mat' % idx)['angle'][0][:N]#.astype(np.int)
    distances = loadmat(DATA_PATH + 'displ_%03d.mat' % idx)['displ'][0][:N]#.astype(np.int)


    if data.shape[0] == 1:
        block = data[0, idx - 1]
    else:
        block = data[idx - 1]


    block = np.atleast_2d(block)
    gt_lines = []
    for line in block.astype(np.int):
        true_x1, true_y1, true_x2, true_y2 = line
        gt_lines += [utils.Line(true_x1 + PAD, true_y1+ PAD, true_x2+ PAD, true_y2+ PAD)]

    #G = 0#len(gt_lines)

    for line in gt_lines:
        line.draw(img_in, white)


    img = color.rgb2gray(img_in)
    mreal, mimag = symmetry.compute_morlet(img, num_angles=32, sigma=2.0)
    w = max(img.shape)

    lines = []
    for k in range(N):
        t = (np.pi/2 - angles[k])%np.pi
        r = distances[k]
        l = utils.InfLine(r, t, img_in)
        x,y = symmetry.comput_center(img,min_dist=0, max_dist=w/2,num_angles=32,
                            morlet_real=mreal, morlet_imag=mimag,
                            r = l.r, angle=l.theta)
        l.cx, l.cy = x,y
        l.draw(img_in, color_list[k])
        lines += [l]


    for i in range(N):
        #print(str(i + 1) + 'Line')
        subset = lines[0:i+1]

        fn = len(gt_lines)
        tp = 0
        fp = 0

        #print('----------------Lines = ' + str(i + 1))

        found = False
        true_lines = list(gt_lines)

        for cur_line in subset:
            for check_line in true_lines:
                dist = check_line.dist_centre_to_centre(cur_line)
                angle_diff_deg = check_line.angle_diff_inf_line_deg(cur_line)
                #print('Thresh = ', 0.2*true_lines[k].len)
                #print('Dist = ', dist)
                #print(dist, angle_diff_deg, dist < 0.2*true_lines[k].len, angle_diff_deg < 10)
                if dist < 0.2*check_line.len and angle_diff_deg < 10:
                    tp += 1
                    fn -= 1
                    #raise IndexError
                    #if tp > 1:
                    #    raise ValueError
                    true_lines.remove(check_line)
                    subset.remove(cur_line)
                    break


        fp = i + 1 - tp

        #print(tp, fp, fn)
        assert tp <= (i + 1)
        assert fp >= 0
        assert fn >= 0
        TP[i] += tp
        FP[i] += fp
        FN[i] += fn
        if (i + 1) == N and tp < 1:
            not_found.append(idx)

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
plt.axes().set_xlim(0,1.1)
plt.axes().set_ylim(0,1.1)
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))

plt.grid(True)

plt.show()
