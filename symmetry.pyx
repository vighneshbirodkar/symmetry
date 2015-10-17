#cython: cdivision=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
import morlet
from skimage import io, util, color
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve

cimport numpy as cnp
from libc.math cimport sqrt, sin, cos, floor, round, abs


cdef cnp.float_t PI_BY_2 = np.pi/2
cdef cnp.float_t PI = np.pi
cdef cnp.float_t TWO_PI = 2*np.pi

cdef inline cnp.float_t adjust_angle(cnp.float_t angle):

    cdef cnp.float_t floor_index = floor(angle/PI)
    return angle - floor_index*PI

def symmetry(img_arr, min_dist, max_dist, morlet_sigma=2.0,morlet_width=16,
             num_angles = 16, debug_flag=False):

    cdef Py_ssize_t phi_idx=0, idx=0, theta_idx=0
    cdef Py_ssize_t xmax, ymax, rho_max, cx, cy, x, y ,x1 ,y1,d, theta_i, theta1_i
    cdef cnp.int_t num_phi = num_angles
    cdef cnp.float_t sigma = morlet_sigma
    cdef cnp.float_t phi = 0, phi_m_pi_by_2
    cdef cnp.float_t delta_theta = 0
    cdef cnp.float_t theta, theta1
    cdef Py_ssize_t rho, rho_min=0
    cdef Py_ssize_t width = 12
    cdef bint debug = debug_flag
    cdef cnp.float_t[::1] phi_list = np.pi*(1.0/num_phi)*np.arange(num_phi)
    #phi_list[:] = 0
    cdef Py_ssize_t num_theta = 3
    cdef cnp.float_t[::1] theta_list = np.array([-np.pi/4, 0, np.pi/4])
    cdef cnp.float_t[:,:,::1] j_real, j_imag
    cdef cnp.float_t[:, ::1] img = np.ascontiguousarray(img_arr)
    cdef cnp.float_t[:, ::1] sym_real
    cdef cnp.float_t[:, ::1] sym_imag
    cdef Py_ssize_t dmin=min_dist, dmax = max_dist
    cdef cnp.float_t ms_real, ms_imag

    d_sym_real_arr = np.zeros(dmax)
    d_sym_imag_arr = np.zeros(dmax)

    d_sym_real = d_sym_real_arr
    d_sym_imag = d_sym_imag_arr

    xmax = img.shape[1]
    ymax = img.shape[0]

    rho_max = <Py_ssize_t>(sqrt(xmax*xmax + ymax*ymax) + 1)

    j_real_arr = np.zeros((ymax, xmax, num_phi), dtype=np.float)
    j_imag_arr = np.zeros((ymax, xmax, num_phi), dtype=np.float)
    j_real = j_real_arr
    j_imag = j_imag_arr

    sym_real_arr = np.zeros((rho_max*2, num_phi))
    sym_imag_arr = np.zeros((rho_max*2, num_phi))

    sym_real = sym_real_arr
    sym_imag = sym_imag_arr


    while idx < num_phi:
        kernel = morlet.kernel(width, sigma, phi_list[idx])
        convolve(img_arr.copy(), np.real(kernel), j_real_arr[:, :, idx])
        convolve(img_arr.copy(), np.imag(kernel), j_imag_arr[:, :, idx])
        #io.imsave('out/real_' + str(idx) + '.png',morlet.normalize(j_real_arr[:, :, idx]) )
        idx += 1

    if debug:
        print('Pre Computing Done')

    phi_idx = 0
    while phi_idx < num_phi:
        phi = phi_list[phi_idx]
        phi_m_pi_by_2 = phi - PI_BY_2

        if debug:
            print('Running for phi Index',phi_idx)

        for cx in range(xmax):

            for cy in range(ymax):

                rho = <Py_ssize_t>(cx*cos(phi - PI_BY_2) + cy*sin(phi - PI_BY_2))


                theta_idx = 0
                while theta_idx < num_theta:

                    delta_theta = theta_list[theta_idx]

                    theta = (phi - PI_BY_2 + delta_theta)%PI
                    theta1 = (2*phi - theta)%PI

                    theta_i = <Py_ssize_t>(theta*num_phi/PI)
                    theta1_i = <Py_ssize_t>(theta1*num_phi/PI)


                    d = dmin
                    while d < dmax:

                        # If `d` could be valid for any angle, `d` should
                        # not be allowed to vote for any angle
                        #if d >= cx or d >= cy or d + cx >= xmax or d + cy >= ymax:
                        #    d += 1
                        #    continue

                        x = <Py_ssize_t>(cx - d*cos(phi_m_pi_by_2))
                        y = <Py_ssize_t>(cy - d*sin(phi_m_pi_by_2))
                        x1 = <Py_ssize_t>(cx + d*cos(phi_m_pi_by_2))
                        y1 = <Py_ssize_t>(cy + d*sin(phi_m_pi_by_2))

                        if x < 0 or y < 0 or x1 < 0 or y1 < 0:
                            d += 1
                            continue

                        if x >= xmax or y >= ymax or x1 >= xmax or y1 >= ymax:
                            d += 1
                            continue

                        ms_real = j_real[y, x, theta_i]*j_real[y1, x1, theta1_i]
                        ms_real += j_imag[y, x, theta_i]*j_imag[y1, x1, theta1_i]

                        ms_imag = -j_real[y, x, theta_i]*j_imag[y1, x1, theta1_i]
                        ms_imag += j_imag[y, x, theta_i]*j_real[y1, x1, theta1_i]

                        sym_real[rho + rho_max, phi_idx] += ms_real
                        sym_imag[rho + rho_max, phi_idx] += ms_imag
                        d += 1

                    theta_idx += 1

        phi_idx += 1


    sym_mag = sym_real_arr**2 + sym_imag_arr**2
    d_mag = d_sym_real_arr**2 + d_sym_imag_arr**2
    return sym_mag, d_mag, np.array(phi_list)


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

    x1,y2,x2,y2 = map(int, (x1,y1,x2,y2))

    return x1,y1,x2,y2
