#!/usr/bin/env python3

import sys
from python_image import *

def main(args):

    a = Img().imread("colors.ppm")
    
    a.imwrite("writtencolors.ppm")
    
    print(a)

    hsi = (a / 255).rgb2hsi()
    
    print("\nhsi:\n\n", hsi)

    rgb = (255*hsi.rgb2hsi())
    
    print("\nrgb:\n\n", rgb)

    
    
if __name__ == "__main__":
    main(sys.argv)
