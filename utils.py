import numpy as np
from skimage.color.rgb_colors import *
from skimage import draw

def angle_diff(a1, a2):
    a1 = a1%180
    a2 = a2%180
    return abs(min(a1 - a2, 180 - (a1 - a2)))


def line_coords(img, sym, dist, angle_bins, drange=10, arange=2, num_lines=1):
    img = img[:,:,0:3]
    #sym = sym.copy()

    lines = []

    for i in range(num_lines):
        r, t = np.unravel_index(np.argmax(sym), sym.shape)
        offset = sym.shape[0]/2
        line = InfLine(r - offset, angle_bins[t], img)

        lines.append(line)

        dmin = np.clip(r - drange - 1, 0, sym.shape[0])
        dmax = np.clip(r + drange + 1, 0, sym.shape[0])
        #dmin = int(round(dmin))
        #dmax = int(round(dmax))
        #print("dmin = ", dmin)
        #print("dmax = ", dmax)

        amin = np.clip(t - arange - 1, 0, sym.shape[1])
        amax = np.clip(t + arange + 1, 0, sym.shape[1])

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
        return abs(num/den)



class Line(object):

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.cx = (x1 + x2)/2
        self.cy = (y1 + y2)/2
        self.angle = np.arctan2(y1 - y2, x1 - x2)
        self.angle = self.angle%np.pi
        self.len = np.hypot(y1 - y2, x1 - x2)

    def __eq__(self, other):
        return (self.x1 == other.x1 and self.y1 == other.y1 and
                self.x2 == other.x2 and self.y2 == other.y2 )

    def dist_to_inf_line(self, line):
        return line.dist_to_point(self.cx, self.cy)

    def angle_diff_inf_line_deg(self, line):
        #print('self angle = ', self.angle*180/np.pi)
        #print('other angle =', (line.theta - np.pi/2)*180/np.pi)
        return angle_diff(self.angle*180/np.pi, (line.theta)*180/np.pi)


class InfLine(object):
    def __init__(self, r, theta, img):
        self.r = r
        self.theta = theta
        line_angle = theta - np.pi/2
        ymax, xmax, ch = img.shape

        if line_angle==0:
            x1 = r
            y1 = 0
            x2 = r
            y2 = ymax-1
        elif line_angle == -np.pi/2:
            x1 = 0
            y1 = -(r)
            x2 = xmax - 1
            y2 = -(r)
        else:
            #line_angle = np.pi - line_angle
            m = -np.cos(line_angle)/np.sin(line_angle)
            c = (r)/np.sin(line_angle)
            # y = mx + c
            x1 = -c/m
            y1 = 0
            x2 = (ymax-1 - c)/m
            y2 = ymax-1

        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def draw(self,img, color=red):

        line = draw.line(self.y1, self.x1, self.y2, self.x2)
        draw.set_color(img, line, color)

    def dist_to_point(self, x, y):
        return abs(dist_point_line(x, y, self.x1, self.y1, self.x2, self.y2))

    def angle_diff_deg(self, angle):
        return abs(angle_diff(self.theta*180/np.pi, angle))

    def __repr__(self):
        msg = 'Line at %d, %d with end points (%d, %d) and (%d, %d)'
        return msg % (self.r, int(self.theta*180/np.pi), self.x1, self.y1, self.x2, self.y2)