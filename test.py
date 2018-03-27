#!/usr/bin/env python3

import sys
from python_image import *

def main(args):
    
    a = Img().imread("images/strawberries.ppm")
    
    print("rgb:\n", a)
    
    # convert original to hsi
    a_hsi = a.rgb2hsi()

    print("hsi:\n", a_hsi)
    
    # get pixels with hue in [0, .05) or (.9, 1] and saturation in (.465, 1]
    hsiind = Img(shape=a[0,:,:].shape)
    hue_ind = np.logical_or(a_hsi[0,:,:] > .7, a_hsi[0,:,:] < .05)
    sat_ind = a_hsi[1,:,:] > .4
    ind = np.logical_and(hue_ind, sat_ind)

    print("ind:\n", ind)
    
    # set all other pixels to gray
    d = a
    d[0,:,:][np.logical_not(ind)] = 128;
    d[1,:,:][np.logical_not(ind)] = 128;
    d[2,:,:][np.logical_not(ind)] = 128;
    
    #d.imshow("result")

    d.imwrite("ppmwrite.ppm")
    d._write_png("pngwrite.png")
    d.imshow("result")
    
if __name__ == "__main__":
    main(sys.argv)
