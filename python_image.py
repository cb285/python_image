#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from subprocess import Popen
from os import setpgrp
import numbers

class ImageReadError(Exception):
    pass

PADTYPE_NONE = "none"
PADTYPE_ZERO = "zero"
PADTYPE_SAME = "same"

class Img():

    """
    Function: __init__
    runs when instance of class is created
    """
    def __init__(self, arg=(0,0)):
        
        # if arg is dimensions, create an empty image of that size
        if(type(arg) is tuple):
            self._size = arg
            self._img_data = np.zeros(self._size, np.uint8)
            self._rep = "gray"
            self._padtype = PADTYPE_NONE
        
        # if arg is a filename, read the image
        elif(type(arg) is str):
            self.imread(arg)
            self._padtype = PADTYPE_NONE
        
        # if arg is another image, copy the image
        elif(type(arg) is type(self)):
            self._size = deepcopy(arg._size) # tuples are immutable, but their elements might not be
            self._img_data = arg._img_data.copy() # arrays are mutable
            self._rep = arg._rep
            self._padtype = PADTYPE_NONE

    """
    Function: __repr__
    defines class's print() and repr() behavior
    """
    def __repr__(self):
        # print the dimensions
        s = "size:" + str(self._size[0]) + "x" + str(self._size[1]) + "\n"
        # print the data type
        s = s + "dtype:" + str(self.dtype()) + "\n"
        # print the representation
        s = s + "rep:" + self._rep + "\n"
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
            if(len(self._size) == 3):
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
                if (len(self._size) == 3):
                    # return three values
                    return self._img_data[:, y, x]
                # if two dimensions
                else:
                    return self._img_data[y, x]
                # if padding is enabled
            else:
                # if out of bounds
                if((y < 0) or (y > self._size[0]) or (x < 0) or (x > self._size[1])):
                    if(self._padtype == PADTYPE_ZERO):
                        return 0
                    elif(self._padtype == PADTYPE_SAME):
                        # if inside x dimensions
                        if(not ((x < 0) or (x > self._size[1]))):
                            # if on top of image
                            if(y < 0):
                                # return closest pixel
                                return(self[0, x])
                            # if on bottom of image
                            else:
                                # return closest pixel
                                return(self[self._size[0] - 1, x])
                            # if inside y dimensions
                        elif(not ((y < 0) or (y > self._size[0]))):
                            # if left of image
                            if(x < 0):
                                # return closest pixel
                                return(self[y, 0])
                            # if to right of image
                            else:
                                # return closest pixel
                                return(self[y, self._size[1] - 1])
                            # if outside both x and y dimensions (corners)
                        else:
                            # top left
                            if(x < 0 and y < 0):
                                # return corner pixel
                                return(self[0,0])
                            # top right
                            elif(y < 0):
                                # return corner pixel
                                return(self[0, self._size[1] - 1])
                            # bottom left
                            elif(x < 0):
                                # return corner pixel
                                return(self[self._size[0] - 1, 0])
                            # bottom left
                            else:
                                # return corner pixel
                                return(self[self._size[0] - 1, self._size[1] - 1])
                # if in bounds
                else:
                    # if image is three dimensions
                    if (len(self._size) == 3):
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
            if (len(self._size) == 3):
                self._img_data[:, y, x] = val
            else:
                self._img_data[y,x] = val

    # addition
    def __add__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data + other
        return new_img
    
    # subtraction
    def __sub__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data - other
        return new_img
    
    # multiplication (self * other)
    def __mul__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data * other
        return new_img
    
    # reverse multiplication (other * self)
    __rmul__ = __mul__

    # division
    def __truediv__(self, other):
        new_img = Img(self)
        new_img._img_data = new_img._img_data // other
        return new_img
    
    # equal
    def __eq__(self, other):
        if(type(other) is Img):
            return np.arrayequal(self._img_data, other._img_data)
        else:
            return False
    
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

    def size(self):
        return self._size
    
    def imshow(self, title):
        # write to temporary file
        filename = title + ".pbm"
        self.imwrite(filename)
        Popen(["nohup", "emacs", filename],
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
    
    def as_type(self, dtype):
        self._img_data = self._img_data.as_type(dtype)
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
        self._size = (m, n)
        self._img_data = np.zeros(self._size, np.uint8)
        self._rep = "bw"
        
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
            for j in range(self._size[0]):
                for i in range(self._size[1]):
                    self._img_data[j, i] = int(data[self._size[1]*j+i])
            
            return self
        
        # binary format
        elif (magic == "P4"):
            
            data = dlines[0]
            
            pixel_num = 0
            byte_num = 0
            shift = 0
            
            # put data into array
            for j in range(self._size[0]):
                for i in range(self._size[1]):
                    pixel_num = self._size[1]*j+i
                    byte_num = int(pixel_num / 8)
                    shift = 7 - pixel_num % 8
                    
                    #print(pixel_num)
                    
                    print((data[byte_num] & (1 << shift)) >> shift)
                    
                    self._img_data[j, i] = ((data[byte_num] & (1 << shift)) >> shift)
                    
            return self
            
        else:
            raise ImageReadError("invalid \"magic number\" for pbm")
        
        return self
    
    def _write_pbm(self, filename):
        
        with open(filename, "wb") as f:
            # write "magic" string
            f.write("P4\n".encode("utf-8"))
            # write size
            f.write((str(self._size[1]) + " " + str(self._size[0]) + "\n").encode("utf-8"))
            # write image data
            np.packbits(self._img_data, axis=-1).tofile(f)
            
        return self

    def _write_pgm(self, filename):
        with open(filename, "wb") as f:
            # write "magic" string
            f.write("P5\n".encode("utf-8"))
            # write size
            f.write((str(self._size[1]) + " " + str(self._size[0]) + "\n").encode("utf-8"))
            # write max value
            f.write("255\n".encode("utf-8"))
            
            # write image data
            self._img_data.tofile(f) #unpackbits(self._img_data, , axis=-1)
            
        return self
    
    def _read_pgm(self, filename):
        with open(filename, "rb") as f:
            flines = f.read().split(bytes(b'\n'))
        
        magic = flines[0].decode("utf-8")
        n, m = map(int, flines[1].decode("utf-8").split(" "))
        max_val = int(flines[2].decode("utf-8"))
        dlines = flines[3:]
        
        # store image info
        self._size = (m, n)
        self._img_data = np.zeros(self._size, np.uint8)
        self._rep = "bw"
        
        # ASCII format
        if(magic == "P2"):
            
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
            for j in range(self._size[0]):
                for i in range(self._size[1]):
                    self._img_data[j, i] = int(255*(float(data[self._size[1]*j+i])/max_val))
            
            return self
        
        # binary format
        elif (magic == "P5"):
            
            data = dlines[0]

            print(len(data))
            
            # put data into array
            for j in range(self._size[0]):
                for i in range(self._size[1]):
                    #print(j, i)
                    self._img_data[j, i] = data[self._size[1]*j+i]
                    
            return self
            
        else:
            raise ImageReadError("invalid \"magic number\" for pbm")
        
        return self
    
    def _read_ppm(self, filename):
        raise ImageReadError("unsupported image format")
