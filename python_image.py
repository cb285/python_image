#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from subprocess import Popen
from os import setpgrp
import numbers
import png

class ImageReadError(Exception):
    pass

PADTYPE_NONE = "none"
PADTYPE_ZERO = "zero"
PADTYPE_SAME = "same"

#IMSHOW_PROGRAM = "imagej"
IMSHOW_PROGRAM = "emacs"


class Img():

    """
    Function: __init__
    runs when instance of class is created
    """
    def __init__(self, arg=(0,0)):
        
        # if arg is dimensions, create an empty image of that size
        if(type(arg) is tuple):
            self.shape = arg
            self._img_data = np.zeros(self.shape, np.uint8)
            self._padtype = PADTYPE_NONE

        # if arg is a filename, read the image
        elif(type(arg) is str):
            self.imread(arg)
            self._padtype = PADTYPE_NONE

        # if arg is another image, copy the image
        elif(type(arg) is type(self)):
            self.shape = deepcopy(arg.shape) # tuples are immutable, but their elements might not be
            self._img_data = arg._img_data.copy() # arrays are mutable
            self._padtype = PADTYPE_NONE
        # if arg is a numpy array
        elif(type(arg) is np.ndarray):
            self.shape = arg.shape
            self._img_data = arg.copy()
            self._padtype = PADTYPE_NONE


    """
    Function: __repr__
    defines class's print() and repr() behavior
    """
    def __repr__(self):
        # print the dimensions
        s = "shape:" + str(self.shape[0]) + "x" + str(self.shape[1]) + "\n"
        # print the data type
        s = s + "dtype:" + str(self.dtype()) + "\n"
        # print the array of data
        s = s + str(self._img_data)
        return s

    """
    Function: __getitem__
    defines indexing when retrieving elements of image (e.g. a[3, 21])
    """
    def __getitem__(self, tup):
        # if index is for three dimensions
        if(len(tup) == 3):
            y, x, z = tup

            # if this image has three dimensions
            if(len(self.shape) == 3):
                # check if padding is enabled
                if(self._padtype == PADTYPE_NONE):
                    return self._img_data[z, y, x]
                else:
                    # check if out of bounds
                    raise Exception("padding unsupported on 3d images")
            # if does not have three dimensions, throw exception
            else:
                raise ValueError("invalid index for a 2d image")

        # if index is for two dimensions
        if(len(tup) == 2):
            y, x = tup
            # if padding is disabled
            if(self._padtype == PADTYPE_NONE):
                # if image is three dimensions
                if (len(self.shape) == 3):
                    # return three values
                    return self._img_data[:, y, x]
                # if two dimensions
                else:
                    return self._img_data[y, x]
                # if padding is enabled
            else:
                # if out of bounds
                if((y < 0) or (y >= self.shape[0]) or (x < 0) or (x >= self.shape[1])):
                    if(self._padtype == PADTYPE_ZERO):
                        return 0
                    elif(self._padtype == PADTYPE_SAME):
                        # if inside x dimensions
                        if(not ((x < 0) or (x >= self.shape[1]))):
                            # if on top of image
                            if(y < 0):
                                # return closest pixel
                                return(self[0, x])
                            # if on bottom of image
                            else:
                                # return closest pixel
                                return(self[self.shape[0] - 1, x])
                            # if inside y dimensions
                        elif(not ((y < 0) or (y >= self.shape[0]))):
                            # if left of image
                            if(x < 0):
                                # return closest pixel
                                return(self[y, 0])
                            # if to right of image
                            else:
                                # return closest pixel
                                return(self[y, self.shape[1] - 1])
                            # if outside both x and y dimensions (corners)
                        else:
                            # top left
                            if(x < 0 and y < 0):
                                # return corner pixel
                                return(self[0,0])
                            # top right
                            elif(y < 0):
                                # return corner pixel
                                return(self[0, self.shape[1] - 1])
                            # bottom left
                            elif(x < 0):
                                # return corner pixel
                                return(self[self.shape[0] - 1, 0])
                            # bottom left
                            else:
                                # return corner pixel
                                return(self[self.shape[0] - 1, self.shape[1] - 1])
                # if in bounds
                else:
                    # if image is three dimensions
                    if (len(self.shape) == 3):
                        # return three values
                        return self._img_data[:, y, x]
                    # if two dimensions
                    else:
                        return self._img_data[y, x]
            
    def __setitem__(self, tup, val):
        if(len(tup) == 3):
            y, x, z = tup
            self._img_data[z, y, x] = val
        
        if(len(tup) == 2):
            y, x = tup
            if (len(self.shape) == 3):
                self._img_data[:, y, x] = val
            else:
                self._img_data[y,x] = val

    # addition
    def __add__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data + other
        return new_img

    __radd__ = __add__
    
    # subtraction
    def __sub__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data - other
        return new_img

    __rsub__ = __sub__
    
    # multiplication (self * other)
    def __mul__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data * other
        return new_img
    
    # reverse multiplication (other * self)
    __rmul__ = __mul__

    # integer division
    # division
    def __floordiv__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data // other
        return new_img

    
    # division
    def __truediv__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data / other
        return new_img

    # equal
    def __eq__(self, other):
        # returns bool array
        result_array = self._img_data == other
        # convert to uint8 array of 1's and 0's
        result_array.astype("uint8")

    def equals(self, other):
        # if both are images
        if(type(other) is Img):
            return np.arrayequal(self._img_data, other._img_data)
        # if other is a numpy array
        elif(type(other) is type(self._img_data)):
            return np.arrayequal(self._img_data, other)
        # otherwise, error
        else:
            raise ValueError("operand must either be an image or numpy array")

    def dtype(self, dt=False):
        if(not dt):
            return self._img_data.dtype
        else:
            self._img_data = self._img_data.astype(dt)
            return self

    def padtype(self, padtype=False):
        # if no argument given, return current padtype
        if(not padtype):
            return self._padtype
        
        if(padtype in [PADTYPE_NONE, PADTYPE_ZERO, PADTYPE_SAME]):
            self._padtype = padtype
        else:
            raise ValueError("invalid argument for padtype")
        
        return self

    def conv2(self, mask, padtype=False):

        # check if mask is valid type
        if(not ((type(mask) is Img) or type(mask) is np.ndarray)):
            raise ValueError("mask must be either an Img or numpy array")

        # check if mask is square
        if(mask.shape[0] != mask.shape[1]):
            raise ValueError("mask must be square")

        # check if mask is of odd size
        if(mask.shape[0] % 2 == 0):
            raise ValueError("mask must be of odd dimensions")

        # if no padtype specified, use image's padtype
        if(not padtype):
            # save original padtype
            orig_padtype = self.padtype()
        else:
            # save original padtype
            orig_padtype = self.padtype()
            # change padtype
            self.padtype(padtype)

        # if padtype is none, raise error
        if(self.padtype() == PADTYPE_NONE):
            raise ValueError("same or zero padding must be enabled for convolution")

        # create copy of image for output
        result_data = np.ndarray(self.shape, self.dtype())

        # get dimensions
        m = self.shape[0]
        n = self.shape[1]

        mask_size = mask.shape[0]
        mask_mid = mask_size // 2

        # perform convolution
        for j in range(m):
            for i in range(n):
                s = 0
                for mj in range(mask_size):
                    for mi in range(mask_size):
                        #print(j - mask_mid + mj, i - mask_mid + mi)
                        #print(mj, mi)
                        s += mask[mj, mi] * self[j - mask_mid + mj, i - mask_mid + mi]
                #print(s)
                result_data[j, i] = s

        # restore original padtype
        self.padtype(orig_padtype)

        # replace image with result
        self._img_data = result_data

        return self

    def imshow(self, title):
        # if three dimensions
        if(len(self.shape) == 3):
            # if is in hsi
            if(self._hsi):
                # convert to rgb
                pass
            # if in rgb
            else:
                write_image = self
                ext = ".ppm"
        # if two dimensions
        elif(len(self.shape) == 2):
            # check if black and white image
            # get count of each unique value
            values, counts = np.unique(self._img_data, return_counts=True)
            # if only two unique values (0 and 1), then write to pbm
            if(len(counts) == 2 and ((values[0] == 1 and values[1] == 0) or (values[0] == 0 and values[1] == 1))):
                ext = ".pbm"
            else:
                ext = ".pgm"
        # otherwise, invalid dimensions
        else:
            raise ValueError("image dimensions are not valid, cannot write")

        # write to file
        filename = title + ext
        self.imwrite(filename)

        # open file
        Popen(["nohup", IMSHOW_PROGRAM, filename],
              stdout=open('/dev/null', 'a'),
              stderr=open('/dev/null', 'a'),
              preexec_fn=setpgrp)
        
        return self

    def imread(self, filename):
        
        # get file extension
        split_filename = filename.split(".")
        ext = split_filename[len(split_filename)-1]
        
        # pbm
        if (ext == "pbm"):
            self._read_pbm(filename)
        # pgm
        elif (ext == "pgm"):
            self._read_pgm(filename)
        # ppm
        elif (ext == "ppm"):
            self._read_ppm(filename)
        else:
            raise ImageReadError("invalid/unsupported image format")
        
        return self

    def imwrite(self, filename):
        
        # get file extension
        split_filename = filename.split(".")
        ext = split_filename[len(split_filename)-1]
        
        # pbm
        if (ext == "pbm"):
            self._write_pbm(filename)
        # pgm
        elif (ext == "pgm"):
            self._write_pgm(filename)
        # ppm
        elif (ext == "ppm"):
            raise ImageReadError("invalid/unsupported image format")
            #self._write_ppm(filename)
        else:
            raise ImageReadError("invalid/unsupported image format")

        return self
    
    def fill(self, val):
        self._img_data.fill(val)
        return self
    
    def _read_pbm(self, filename):
        
        with open(filename, "rb") as f:
            flines = f.read().split(bytes(b'\n'))
        
        magic = flines[0].decode("utf-8")
        n, m = map(int, flines[1].decode("utf-8").split(" "))
        dlines = flines[2:]

        # store image info
        self.shape = (m, n)
        self._img_data = np.zeros(self.shape, np.uint8)
        
        # ASCII format
        if(magic == "P1"):
            
            data = list()
            
            for line in dlines:
                line_words = line.decode("utf-8").split(" ")
                for w in line_words:
                    # remove special characters
                    w.strip()
                    # skip comments and empty words
                    if(len(w) == 0):
                        continue
                    elif(w[0] == "#"):
                        break
                    data.append(w.strip())
            
            # put data into array
            for j in range(self.shape[0]):
                for i in range(self.shape[1]):
                    self._img_data[j, i] = 1 - int(data[self.shape[1]*j+i])
            
            return self
        
        # binary format
        elif (magic == "P4"):
            
            data = dlines[0]
            
            pixel_num = 0
            byte_num = 0
            shift = 0
            
            # put data into array
            for j in range(self.shape[0]):
                for i in range(self.shape[1]):
                    pixel_num = self.shape[1]*j+i
                    byte_num = int(pixel_num / 8)
                    shift = 7 - pixel_num % 8
                    
                    self._img_data[j, i] = ((data[byte_num] & (1 << shift)) >> shift)
                    
            return self
            
        else:
            raise ImageReadError("invalid \"magic number\" for pbm")
        
        return self
    
    def _write_pbm(self, filename):

        # invert values
        write_data = 1 - self._img_data

        with open(filename, "wb") as f:
            # write "magic" string
            f.write("P4\n".encode("utf-8"))
            # write size
            f.write((str(self.shape[1]) + " " + str(self.shape[0]) + "\n").encode("utf-8"))
            # write image data
            np.packbits(self._img_data, axis=-1).tofile(f)

        return self

    def _write_pgm(self, filename):
        with open(filename, "wb") as f:
            # write "magic" string
            f.write("P5\n".encode("utf-8"))
            # write size
            f.write((str(self.shape[1]) + " " + str(self.shape[0]) + "\n").encode("utf-8"))
            # write max value
            f.write("255\n".encode("utf-8"))
            
            # write image data
            self._img_data.tofile(f) #unpackbits(self._img_data, , axis=-1)
            
        return self
    
    def _read_pgm(self, filename):
        f = open(filename, "rb")

        # read header
        magic = f.readline().decode("utf-8").strip()
        n, m = map(int, f.readline().decode("utf-8").strip().split(" "))
        max_val = int(f.readline().decode("utf-8").strip())

        # store image info
        self.shape = (m, n)
        self._img_data = np.zeros(self.shape, np.uint8)

        # ASCII format
        if(magic == "P2"):
            
            data = list()

            dlines = f.readlines()
            
            for line in dlines:
                line_words = line.decode("utf-8").split(" ")
                for w in line_words:
                    # remove special characters
                    w.strip()
                    # skip comments and empty words
                    if(len(w) == 0):
                        continue
                    elif(w[0] == "#"):
                        break
                    data.append(w.strip())
            
            # put data into array
            for j in range(self.shape[0]):
                for i in range(self.shape[1]):
                    self._img_data[j, i] = int(255*(float(data[self.shape[1]*j+i])/max_val))
        
        # binary format
        elif (magic == "P5"):
            # put data into array
            for j in range(self.shape[0]):
                for i in range(self.shape[1]):
                    self._img_data[j, i] = int.from_bytes(f.read(1), byteorder="big")
            
        else:
            raise ImageReadError("invalid \"magic number\" for pbm")

        f.close()
        return self
    
    def _read_ppm(self, filename):
        raise ImageReadError("unsupported image format")
