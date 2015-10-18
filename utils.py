import numpy as np

def angle_diff(a1, a2):
    a1 = a1%180
    a2 = a2%180
    return min(a1 - a2, 180 - (a1 - a2))


def line_coords(img, sym, dist, angle_bins, drange=10, arange=1, num_lines=1):
    img = img[:,:,0:3]
    #sym = sym.copy()

    lines = []

    for i in range(num_lines):
        r, t = np.unravel_index(np.argmax(sym), sym.shape)
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

        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        lines.append((x1,y1,x2,y2,r,t))

        dmin = np.clip(r - drange - 1, 0, sym.shape[0])
        dmax = np.clip(r + drange + 1, 0, sym.shape[0])
        #print("dmin = ", dmin)
        #print("dmax = ", dmax)

        amin = np.clip(t - arange, 0, sym.shape[1])
        amax = np.clip(t + arange, 0, sym.shape[1])

        sym[dmin:dmax, amin:amax] = 0

    return lines


def dist_point_line(x,y,x1,y1,x2,y2):

    if y1 == y2:
        return y - y1
    elif x1 == x2:
        return x - x1
    else:
        #m = float(y2 - y1)/(x2 - x1)
        #c = y1 - m*x1
        #dist = (y - m*x -c)/np.sqrt(1 + m*m)
        num = (y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1
        den = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return num/den
