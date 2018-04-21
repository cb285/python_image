#!/usr/bin/env python3

import sys
from python_image import *

def main(args):
    
    img = imread("images/moon.png")

    original = img.copy()
    
    print(img.shape)

    """
    mask = Img((3,3), dtype=float)

    mask[0,0] = 0
    mask[0,1] = 1
    mask[0,2] = 0
    mask[1,0] = 1
    mask[1,1] = -4
    mask[1,2] = 1
    mask[2,0] = 0
    mask[2,1] = 1
    mask[2,2] = 0
    mask = -1*mask
    """

    mask = Img((3,3))
    mask.fill(1)
    mask = mask / 9

    print(mask)

    img = img.astype(float)
    
    img.conv2(mask, PADTYPE_SAME)

    img_max = np.amax(img)
    img_min = np.amin(img)

    img = 255*((img - img_min) / (img_max - img_min))

    img = img.astype(np.uint8)

    original.imshow(figure=1, block=False)
    img.imshow(figure=2, block=True)

    print(img.dtype)
    
    img.imwrite("conv_image.png")

    print(img[:])

if __name__ == "__main__":
    main(sys.argv)
