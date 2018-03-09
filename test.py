#!/usr/bin/env python3

import sys
from python_image import *

def main(args):

    a = Img().imread("colors.ppm")

    print(a)
    
    #a.imwrite("writetest_colors.ppm")    
    
if __name__ == "__main__":
    main(sys.argv)
