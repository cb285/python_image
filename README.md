# python_image
Simple image processing library for python

# Setup
## Install dependencies
`# pip3 install numpy`

`# pip3 install matplotlib`

## Clone repository
`git clone https://github.com/cb285/python_image/`

## Import
`from python_image import *` or `import python_image`

# Numpy inheritance
python_image's Img type inherits from numpy's ndarray, which allows for compatability with a large number of complex functions provided by numpy. more information can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html).

if you need to convert between an Img type and ndarray, use the view function:

Img to ndarray: `img.view(np.ndarray)`

ndarray to Img: `img.view(Img)`

# Reading image files
`imread(filename)`

Reads a file into an image.

Parameters:
- `filename`: string, name of image file to read

Returns:
- Img, image created by reading file

Example: `img = imread("strawberries.png")`

Supported file types:
- png
- NetPBM Formats (pbm, pgm, ppm)

# Writing image files
`Img.imwrite(filename)`

Writes an image to file.

Parameters:
- `filename`: string, name of file to create

Filename extension specifies filetype. Supports same file types as reading.

Example: `img.imwrite("my_image.png")`

# Displaying images
`Img.imshow(figure, block)`

Displays an image in a window.

Parameters:
- `figure`: integer, figure number - change for displaying multiple images at one time (default: 1)
- `block`: bool, if True then function blocks until the window is closed (default: True)

Example: `img.imshow(1, False)`

Note: if blocking is disabled then all windows will close when the program finishes.

# New image creation
`Img(shape, dtype)`

Creates a new image with specified shape and data type initially filled with zeros.

Parameters:
- `shape`: tuple, shape (dimensions) of created array (default: (1, 1))
- `dtype`: data-type (python or numpy), datatype of created image (default: uint8)

Returns:
- Img, new image filled with zeros

# Pixel addressing
## 2D image
`Img[y, x]`
- `y`: vertical distance from top of image
- `x`: horizontal distance from left of image

## 3D image
`Img[z, y, x]`
- `z`: plane
  - 0 : Red/Hue
  - 1 : Green/Saturation
  - 2 : Blue/Intensity
- `y`: vertical distance from top of image
- `x`: horizontal distance from left of image

## Getting pixel value
Example: `value = img[4, 30]`

## Setting pixel value
Example: `img[43, 12] = value`

# Padding
## Setting pad type:
`Img.padtype(padtype)`

Parameters:
- `padtype`: padtype, string constant specifying desired pad type

## Getting current pad type
`Img.padtype()`

Returns the current pad type of the image as a string constant.

Returns:
- string constant, current padtype

Pad types:

- `PADTYPE_NONE`: padding disabled, out-of-bounds addressing will cause an exception (default)
- `PADTYPE_ZERO`: out-of-bounds addressing will return zero
- `PADTYPE_SAME`: out-of-bounds addressing will return the nearest pixel's value

# RGB & HSI conversion
`Img.rgb2hsi()`

Converts the image from RGB to HSI representation.

Returns:
- Img, hsi representation of RGB image

Example: `hsi = rgb_img.rgb2hsi()`


`Img.hsi2rgb()`

Converts the image from HSI to RGB representation.

Returns:
- Img, rgb representation of HSI image

# 2D convolution
`Img.conv2(mask, padtype)`

Performs in-place 2D convolution. Values will be in the range `[0, 1]`

Parameters:
- `mask`: Img, odd-dimensioned image to use as mask
- `padtype`: string constant, specifies padding to be used during convolution (defaults to image's current pad type)
