#!/usr/bin/env python3

import sys
from python_image import *

def main(args):
    
    # create empty image
    a = Img()

    # read pgm image
    a.imread('feep.pgm')

    # write binary pgm image
    a.imwrite("binary_feep.pgm")
    
    # create 3x3 image
    b = Img((3,3))
    
    for i in range(3):
        for j in range(3):
            b[j,i] = j*3+i

    print(b)
    
    # no padding
    #print(b[4,2])
    
    # zero padding
    print("zero padding:")
    b.padtype(PADTYPE_ZERO)
    print("b[6,8] = " + str(b[6,8]))
    
    # same padding
    print("same padding")
    b.padtype(PADTYPE_SAME)
    print("b[6,8] = " + str(b[6,8]))
    print("b[8,1] = " + str(b[8,1]))

    print("\n")

    # read binary pgm
    a.imread("binary_feep.pgm")
    
    # create image 3 times size of a
    a_size = a.size()
    mag = 5
    c = Img((a_size[0]*mag,a_size[1]*mag))
    
    c_size = c.size()

    print("a_size = " + str(a_size))
    print("c_size = " + str(c_size))
    
    for j in range(c_size[0]):
        for i in range(c_size[1]):
            y = round(j / mag)
            x = round(i / mag)
            
            if(y > a_size[0] - 1):
                y = a_size[0] - 1

            if(x > a_size[1] - 1):
                x = a_size[1] - 1

            c[j,i] = a[y, x]
    
    # write to file
    c.imwrite("feep_resized.pgm")
    
if __name__ == "__main__":
    main(sys.argv)
