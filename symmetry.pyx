#cython: cdivision=False
#cython: nonecheck=False
#cython: boundscheck=True
#cython: wraparound=False

import numpy as np
import morlet
from skimage import io, util, color
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve
from cython.parallel import prange

cimport numpy as cnp
from libc.math cimport sqrt, sin, cos, floor, round, abs, log


cdef cnp.float_t PI_BY_2 = np.pi/2
cdef cnp.float_t PI = np.pi
cdef cnp.float_t TWO_PI = 2*np.pi

cdef inline cnp.float_t adjust_angle(cnp.float_t angle):

    cdef cnp.float_t floor_index = floor(angle/PI)
    return angle - floor_index*PI


def compute_morlet(img_arr, num_angles=16, width=16, sigma=2.0):


    angles = np.pi*(1.0/num_angles)*np.arange(num_angles)
    j_real_arr = np.zeros((img_arr.shape[0], img_arr.shape[1], num_angles), dtype=np.float)
    j_imag_arr = np.zeros((img_arr.shape[0], img_arr.shape[1], num_angles), dtype=np.float)

    for idx in range(num_angles):
        angle = angles[idx]
        kernel = morlet.kernel(width, sigma, angle)
        convolve(img_arr.copy(), np.real(kernel), j_real_arr[:, :, idx])
        convolve(img_arr.copy(), np.imag(kernel), j_imag_arr[:, :, idx])

    return j_real_arr, j_imag_arr

def symmetry(img_arr, min_dist, max_dist, morlet_real, morlet_imag,
             num_angles, debug_flag=False):

    cdef Py_ssize_t phi_idx, idx=0, theta_idx=0
    cdef Py_ssize_t xmax, ymax, rho_max, cx, cy, x, y ,x1 ,y1,d, theta_i, theta1_i
    cdef cnp.int_t num_phi = num_angles
    cdef cnp.float_t phi = 0, phi_m_pi_by_2
    cdef cnp.float_t delta_theta = 0
    cdef cnp.float_t theta, theta1
    cdef Py_ssize_t rho, rho_min=0
    cdef Py_ssize_t width = 12
    cdef bint debug = debug_flag
    cdef cnp.float_t[::1] phi_list = np.pi*(1.0/num_phi)*np.arange(num_phi)
    #phi_list[:] = 0
    cdef Py_ssize_t num_theta = 5
    cdef cnp.float_t[::1] theta_list = np.array([-np.pi/3,-np.pi/6, 0, np.pi/6,np.pi/3])
    cdef cnp.float_t[:,:,::1] j_real, j_imag
    cdef cnp.float_t[:, ::1] img = np.ascontiguousarray(img_arr)
    cdef cnp.float_t[:, ::1] sym_real
    cdef cnp.float_t[:, ::1] sym_imag
    cdef cnp.float_t sym_max,
    cdef Py_ssize_t dmin=min_dist, dmax = max_dist
    cdef cnp.float_t ms_real, ms_imag
    cdef cnp.float_t[:,::1] debug_map = np.zeros_like(img_arr, dtype=np.float)


    xmax = img.shape[1]
    ymax = img.shape[0]

    rho_max = <Py_ssize_t>(sqrt(xmax*xmax + ymax*ymax) + 1)

    j_real = morlet_real
    j_imag = morlet_imag

    sym_real_arr = np.zeros((rho_max*2, num_phi))
    sym_imag_arr = np.zeros((rho_max*2, num_phi))

    sym_real = sym_real_arr
    sym_imag = sym_imag_arr

    if debug:
        print('Pre Computing Done')

    for phi_idx in range(num_phi):
        phi = phi_list[phi_idx]
        phi_m_pi_by_2 = phi - PI_BY_2

        for cx in range(xmax):

            for cy in range(ymax):

                rho = <Py_ssize_t>(cx*cos(phi - PI_BY_2) + cy*sin(phi - PI_BY_2))

                d = dmin
                while d < dmax:


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


                    theta_idx = 0
                    sym_max = 0
                    while theta_idx < num_theta:

                        delta_theta = theta_list[theta_idx]

                        theta = (phi - PI_BY_2 + delta_theta)%PI
                        theta1 = (phi + PI_BY_2 - delta_theta)%PI

                        theta_i = <Py_ssize_t>(theta*num_phi/PI)
                        theta1_i = <Py_ssize_t>(theta1*num_phi/PI)



                        # If `d` could be valid for any angle, `d` should
                        # not be allowed to vote for any angle
                        #if d >= cx or d >= cy or d + cx >= xmax or d + cy >= ymax:
                        #    d += 1
                        #    continue


                        ms_real = j_real[y, x, theta_i]*j_real[y1, x1, theta1_i]
                        ms_real += j_imag[y, x, theta_i]*j_imag[y1, x1, theta1_i]

                        ms_imag = -j_real[y, x, theta_i]*j_imag[y1, x1, theta1_i]
                        ms_imag += j_imag[y, x, theta_i]*j_real[y1, x1, theta1_i]

                        sym_real[rho + rho_max, phi_idx] += ms_real#/log(1 + d)
                        sym_imag[rho + rho_max, phi_idx] += ms_imag#/log(1 + d)

                        #if (rho + rho_max ) == 241 and phi_idx == 10:
                        #    debug_map[y1, x1] =  ms_real**2 + ms_imag**2
                        #    debug_map[y, x] = ms_real**2 + ms_imag**2

                        theta_idx += 1

                    d += 1



    #io.imsave('/home/vighnesh/Desktop/debug.png', morlet.normalize(np.array(debug_map)))

    #plt.imshow(np.array(debug_map), cmap='gray')
    #plt.show()
    sym_mag = sym_real_arr**2 + sym_imag_arr**2
    return sym_mag, np.array(phi_list)

def comput_center(img_arr, min_dist, max_dist, morlet_real, morlet_imag,
                  num_angles, r, angle):

    cdef Py_ssize_t t, theta_idx, theta1_idx
    cdef Py_ssize_t xmax = img_arr.shape[1]
    cdef Py_ssize_t ymax = img_arr.shape[0]
    cdef Py_ssize_t rho_max = <Py_ssize_t>(sqrt(xmax*xmax + ymax*ymax) + 1)
    cdef Py_ssize_t cx, cy
    cdef Py_ssize_t x, y
    cdef Py_ssize_t rho = <Py_ssize_t>r
    cdef cnp.float_t phi = angle
    cdef Py_ssize_t num_phi = num_angles
    cdef cnp.float_t[:,::1] debug_img = np.zeros_like(img_arr, dtype=np.float)
    cdef cnp.float_t weighted_sum = 0
    cdef cnp.float_t weight_sum = 0
    cdef Py_ssize_t dmin = min_dist
    cdef Py_ssize_t dmax = max_dist
    cdef Py_ssize_t d
    cdef Py_ssize_t num_theta=3
    cdef cnp.float_t[::1] theta_list = np.array([-np.pi/4, 0, np.pi/4])
    cdef cnp.float_t theta, theta1, delta_theta
    cdef cnp.float_t[:,:,::1] j_real, j_imag
    cdef cnp.float_t ms_real, ms_imag
    cdef cnp.float_t weight, avg_t, start, end
    cdef cnp.float_t[::1] weights
    cdef cnp.float_t[::1] phi_list = np.array([angle - np.pi/16, angle, angle + np.pi/16])

    weight_array = np.zeros(rho_max*2)
    weights = weight_array

    j_real = morlet_real
    j_imag = morlet_imag

    cx = <Py_ssize_t>(rho*cos(phi - PI_BY_2))
    cy = <Py_ssize_t>(rho*sin(phi - PI_BY_2))

    for t in range(-rho_max, rho_max):
        x = <Py_ssize_t>(cx - t*cos(PI - phi))
        y = <Py_ssize_t>(cy + t*sin(PI - phi))

        for theta_idx in range(num_theta):
            delta_theta = theta_list[theta_idx]

            theta = (phi - PI_BY_2 + delta_theta)%PI
            theta1 = (2*phi - theta)%PI

            theta_i = <Py_ssize_t>(theta*num_phi/PI)
            theta1_i = <Py_ssize_t>(theta1*num_phi/PI)

            for d in range(dmin, dmax):
                x1 = <Py_ssize_t>(x - d*cos(theta - PI_BY_2))
                y1 = <Py_ssize_t>(y - d*sin(theta - PI_BY_2))
                x2 = <Py_ssize_t>(x + d*cos(theta - PI_BY_2))
                y2 = <Py_ssize_t>(y + d*sin(theta - PI_BY_2))

                if x1 >= xmax or y1 >= ymax or x2 >= xmax or y2 >= ymax:
                    continue

                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue

                ms_real = j_real[y1, x1, theta_i]*j_real[y2, x2, theta1_i]
                ms_real += j_imag[y1, x1, theta_i]*j_imag[y2, x2, theta1_i]

                ms_imag = -j_real[y1, x1, theta_i]*j_imag[y2, x2, theta1_i]
                ms_imag += j_imag[y1, x1, theta_i]*j_real[y2, x2, theta1_i]

                weight = ms_real*ms_real + ms_imag*ms_imag
                weights[t + rho_max] = weight #+ .7/(d*d)
                #weighted_sum += t*weight
                #weight_sum += weight
                debug_img[y1, x1] += weight
                debug_img[y2, x2] += weight


    weight_array = np.sqrt(weight_array)
    t_array = np.arange(weight_array.shape[0])
    avg_t = np.sum(weight_array*t_array)/np.sum(weight_array) - rho_max
    #plt.figure()
    x = <Py_ssize_t>(cx - avg_t*cos(PI - phi))
    y = <Py_ssize_t>(cy + avg_t*sin(PI - phi))


    #plt.plot(weight_array)
    #plt.show()

    n = weight_array.shape[0]



    #print(x,y)
    return int(x), int(y)
    #plt.imshow(np.array(debug_img), cmap='gray')
    #plt.show()
