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

if for some reason you need to convert between an Img type and ndarray, use the view function:

Img to ndarray: `img.view(np.ndarray)`

ndarray to Img: `img.view(Img)`

# Image File Reading
`imread(filename)`

Parameters:
- filename: string, name of image file to read

Returns:
- Img, image created by reading file

Example: `img = imread("strawberries.png")`

Supported file types:
- png
- NetPBM Formats (pbm, pgm, ppm)

# Image File Writing
`Img.imwrite(filename)`

parameters:
- filename: string, name of file to create

Filename extension specifies filetype. Supports same file types as reading.

Example: `img.imwrite("my_image.png")`

# Displaying Images
`Img.imshow(figure, block)`

Parameters:
- figure: integer, figure number - change for displaying multiple images at one time (default: 1)
- block: bool, if True then function blocks until the window is closed (default: True)

Example: `img.imshow(1, False)`

Note: if blocking is disabled then all windows will close when the program finishes.

# New Image Creation
`Img(shape, dtype)`

Parameters:
- shape: tuple, shape (dimensions) of created array (default: (1, 1))
- dtype: data-type (python or numpy), datatype of created image (default: uint8)

Returns:
- Img, new image filled with zeros

# Pixel addressing
## Getting pixel value
Example: `value = img[4, 30]`

## Setting pixel value
Example: `img[43, 12] = value`

# Padding
## Setting pad type:
`Img.padtype(padtype)`

Parameters:
- padtype: padtype, string constant specifying desired pad type

## Getting current pad type
`Img.padtype()`

Returns:
- string constant, current padtype

Pad types:

- `PADTYPE_NONE`: padding disabled, out-of-bounds addressing will cause an exception (default)
- `PADTYPE_ZERO`: out-of-bounds addressing will return zero
- `PADTYPE_SAME`: out-of-bounds addressing will return the nearest pixel's value

rbg hsi conversion
2D convolution
uint8 rollover