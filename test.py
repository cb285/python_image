#!/usr/bin/env python3

import sys
from python_image import *

def main(args):

    # read original image
    a = Img().imread('woman.pgm')

    a.imshow("original")

    a.dtype("float")

    # create averaging mask
    mask = Img((5,5)).fill(1) / 25

    # print mask
    print("mask:")
    print(mask)

    # create b as a copy of a
    b = Img(a)

    # convolution
    b.conv2(mask, PADTYPE_SAME)

    a.dtype("uint8")
    b.dtype("uint8")

    # show result
    b.imshow("result")

if __name__ == "__main__":
    main(sys.argv)
