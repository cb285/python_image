# python_image
Simple image processing library for python

Planned Features:

image reading/writing:
      supported filetypes: pbm, ppm, pgm, tif, png
      pbm, ppm, pgm implemented in python (or C/C++) (https://docs.python.org/3.4/extending/extending.html)
      png implemented in C/C++ (libpng)
      functions:
	img.imread(filename)
	img.imwrite(filename)

displaying images:
	   implemented by calling external program (JImage?)
  	   functions:
	img.imshow(title)
	
image types:
	gray (2x2), rgb (2x2x3), hsi (2x2x3)
	uint8 (0-255), float (0-1), double (0-1)
	images stored as array (numpy array?)
	functions:
		img.type() - returns image type as string
		img.type(IMG_TYPE) - cast to type

indexing:
	single pixels: img.at(x,y) - get/set value
	arrays of bools: ind = img > a

padding:
	functions: img.pad_mode(PAD_MODE)
	modes: constant, replicate

looping:
	over entire image (img.for_xy)
	over neighborhood (img.for_nbhood)

convolution:
	mask stored as image/array
	functions: img.conv2(mask) - in place convolution
	
fft2/ifft2:
	implemented using numpy
	functions: img.fft2, img.ifft2
  
