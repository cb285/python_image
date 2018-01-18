#!/usr/bin/env python3

import sys
from python_image import Img

def main(args):
    
    # create empty image
    #a = Img("sierpinski.ascii.pbm")
    a = Img((10,10)).fill(1)
    a.imwrite("writetest.pbm")

    b = Img("writetest.pbm")
    
if __name__ == "__main__":
    main(sys.argv)
