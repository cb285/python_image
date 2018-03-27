#!/usr/bin/env python3

"""
TODO:

- fix padding
- fft2, ifft2 (with padding)
- reading/writing png files
- prevent uint8 rollover (?)

"""

import numpy as np
from math import cos, acos, sqrt, pow, radians
from copy import deepcopy
from subprocess import Popen
from os import setpgrp
import numbers
import png

PADTYPE_NONE = "none"
PADTYPE_ZERO = "zero"
PADTYPE_SAME = "same"

IMSHOW_PROGRAM = "imagej"
#IMSHOW_PROGRAM = "emacs"

class Img(np.ndarray):
    
    def __new__(subtype, shape=(1,1), dtype=np.uint8, buffer=None, offset=0,
                strides=None, order=None):
        
        obj = super(Img, subtype).__new__(subtype, shape, dtype,
                                          buffer, offset, strides,
                                          order)
        
        # fill array with zeros
        obj.fill(0)
        
        # return newly created instance
        return obj

    """
    def __getitem__(self, index):

        #print("index = ", index)
        
        try:
            return super(Img, self).__getitem__(index)
        
        except IndexError:

            # if padding is disabled, re-raise exception
            if(self._padtype == PADTYPE_NONE):
                raise

            # if pad type is zero, return zero
            if(self._padtype == PADTYPE_ZERO):
                return (self.dtype)(0)
            
            # index == (z, y, x)

            # if index has one dimension
            if(len(index) == 1):
                z = index[0]

                if(z < 0):
                    return self[0]
                else:
                    return self[self.shape[0] - 1]

            # if index has two dimensions
            if(len(index) == 2):

                y, x = index
                
                y, x = self._get_xy_same_ind(y, x)

                print(y,x)
                
                return self[y, x]

            if(len(index) == 3):

                z, y, x = index

                y, x = self._get_xy_same_ind(y, x)
                
                if(z < 0):
                    return self[0, y, x]
                elif(z >= self.shape[0]):
                    return self[self.shape[0] - 1, y, x]
                else:
                    return self[z, y, x]
    """

    def _get_xy_same_ind(self, y, x):

        print("index =", y, x)
        
        # top
        if(y < 0):
            # left
            if(x < 0):
                return 0, 0
            # right
            elif(x >= self.shape[1]):
                return 0, self.shape[1] - 1
            # inside
            else:
                return 0, x
            
        # bottom
        elif(y >= self.shape[0]):
            # left
            if(x < 0):
                return self.shape[0] - 1, 0
            # right
            elif(x >= self.shape[1]):
                return self.shape[0] - 1, self.shape[1] - 1
            # inside
            else:
                return self.shape[0] - 1, x

        # left
        elif(x < 0):
            return y, 0
        # right
        elif(x >= self.shape[1]):
            return y, self.shape[1] - 1
        else:
            return -1, -1

    def padtype(self, padtype=False):
        # if no argument given, return current padtype
        if(not padtype):
            return self._padtype
        
        if(padtype in [PADTYPE_NONE, PADTYPE_ZERO, PADTYPE_SAME]):
            self._padtype = padtype
        else:
            raise ValueError("invalid argument for padtype")

        return self

    def rgb2hsi(self):

        if(self.dtype == np.uint8):
            float_img = self.astype(np.float)
            float_img = float_img / 255
        
        r = float_img[0,:,:]
        g = float_img[1,:,:]
        b = float_img[2,:,:]
        
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt(((r - g)*(r - g)) + (r - b)*(g - b))
        
        ind = np.zeros(shape=r.shape, dtype=bool)
        theta = np.zeros(shape=r.shape, dtype=float)

        ind = (den != 0)

        h = np.zeros(shape=r.shape, dtype=float)
        
        h[ind] = np.arccos(num[ind] / den[ind]) / (2*np.pi)

        ind = b > g
        
        h[ind] = 1 - theta[ind]

        # calculate intensity
        i = (r + g + b) / 3
        
        s = np.zeros(shape=r.shape, dtype=float)

        # calculate saturation
        ind = i != 0
        s[ind] = 1 - np.minimum(np.minimum(r, g), b)[ind] / i[ind]

        # return in new image
        hsi = Img(shape=self.shape, dtype=float)
        
        hsi[0,:,:] = h
        hsi[1,:,:] = s
        hsi[2,:,:] = i

        return hsi

    def hsi2rgb(self):
        
        h = self[0,:,:]*2*np.pi
        s = self[1,:,:]
        i = self[2,:,:]

        r = np.zeros(shape=h.shape, dtype=float)
        g = np.zeros(shape=h.shape, dtype=float)
        b = np.zeros(shape=h.shape, dtype=float)

        # RG sector
        ind = np.logical_and(0 <= h, h < radians(120))
        b[ind] = i[ind]*(1 - s[ind])
        r[ind] = i[ind]*(1 + (s[ind]*np.cos(h[ind])) / np.cos(radians(60) - h[ind]))
        g[ind] = 3*i[ind] - (r[ind] + b[ind])

        # GB sector
        ind = np.logical_and(radians(120) <= h, h < radians(240))
        h_gb = h - radians(120)
        r[ind] = i[ind]*(1 - s[ind])
        g[ind] = i[ind]*(1 + (s[ind]*np.cos(h_gb[ind]) / np.cos(radians(60) - h_gb[ind])))
        b[ind] = 3*i[ind] - (r[ind] + g[ind])
        
        # RB sector
        ind = np.logical_and(radians(240) <= h, h <= radians(360))
        h_rb = h - radians(240)
        g[ind] = i[ind]*(1 - s[ind])
        b[ind] = i[ind]*(1 + (s[ind]*np.cos(h_rb[ind]) / np.cos(radians(60) - h_rb[ind])))
        r[ind] = 3*i[ind] - (g[ind] + b[ind])

        # return in new image
        rgb = Img(shape=self.shape, dtype=float)
        
        rgb[0,:,:] = r
        rgb[1,:,:] = g
        rgb[2,:,:] = b

        return rgb
    
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
            write_image = self
            ext = ".ppm"
        # if two dimensions
        elif(len(self.shape) == 2):
            # if it is bool image, write to b/w
            if(self._img_data.dtype is bool):
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
        # png
        elif(ext == "png"):
            self._read_png(filename)
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
            self._write_ppm(filename)
        elif(ext == "png"):
            self._write_png(filename)
        else:
            raise ImageReadError("invalid/unsupported image format")

        return self

    def _read_png(self, filename):
        r = png.Reader(filename=filename).read()

        print(r[0])

    def _write_png(self, filename):

        with open(filename, 'wb') as f:
            rows, cols, planes = self.shape
            writer = png.Writer(cols, rows, greyscale=False, alpha=False, bitdepth=8)
            writer.write(f, np.reshape(np.rot90(self), (-1, cols*planes)))

    def _read_pbm(self, filename):
        
        with open(filename, "rb") as f:
            flines = f.read().split(bytes(b'\n'))
        
        magic = flines[0].decode("utf-8")
        n, m = map(int, flines[1].decode("utf-8").split(" "))
        dlines = flines[2:]

        self.resize((m, n), refcheck=False)
        self.dtype = np.uint8

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
                    self[j, i] = 1 - bool(data[self.shape[1]*j+i])
            
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
                    
                    self[j, i] = bool((data[byte_num] & (1 << shift)) >> shift)
                    
            return self
            
        else:
            raise ImageReadError("invalid \"magic number\" for pbm")
        
        return self
    
    def _write_pbm(self, filename):

        # invert values
        write_data = 1 - self

        with open(filename, "wb") as f:
            # write "magic" string
            f.write("P4\n".encode("utf-8"))
            # write size
            f.write((str(self.shape[1]) + " " + str(self.shape[0]) + "\n").encode("utf-8"))
            # write image data
            np.packbits(self, axis=-1).tofile(f)

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
            self.tofile(f) #unpackbits(self._img_data, , axis=-1)
            
        return self
    
    def _read_pgm(self, filename):
        f = open(filename, "rb")

        # read header
        magic = f.readline().decode("utf-8").strip()
        n, m = map(int, f.readline().decode("utf-8").strip().split(" "))
        max_val = int(f.readline().decode("utf-8").strip())

        self.resize((m, n), refcheck=False)
        self.dtype = np.uint8

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
                    self[j, i] = int(255*(float(data[self.shape[1]*j+i])/max_val))
        
        # binary format
        elif (magic == "P5"):
            # put data into array
            for j in range(self.shape[0]):
                for i in range(self.shape[1]):
                    self[j, i] = int.from_bytes(f.read(1), byteorder="big")

        else:
            raise ImageReadError("invalid \"magic number\" for pbm")

        f.close()
        return self
    
    def _read_ppm(self, filename):
        f = open(filename, "rb")

        # read header
        magic = f.readline().decode("utf-8").strip()
        n, m = map(int, f.readline().decode("utf-8").strip().split(" "))
        max_val = int(f.readline().decode("utf-8").strip())

        self.resize((3, m, n), refcheck=False)
        self.dtype = np.uint8

        # ASCII format
        if(magic == "P3"):
            
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

            pixel_num = 0

            # put data into array
            for j in range(self.shape[1]):
                for i in range(self.shape[2]):
                    for p in range(3):
                        #print("img[",p,",",j,",",i,"] = ", int(255*(float(data[pixel_num])/max_val)))
                        self[p, j, i] = int(255*(float(data[pixel_num])/max_val))
                        pixel_num += 1
        
        # binary format
        elif (magic == "P6"):
            # put data into array
            for j in range(self.shape[1]):
                for i in range(self.shape[2]):
                    for p in range(3):
                        self[p, j, i] = int(255*float(int.from_bytes(f.read(1), byteorder="big")) / max_val)
            
        else:
            raise ImageReadError("invalid \"magic number\" for ppm")
        
        f.close()
        return self

    def _write_ppm(self, filename):
        with open(filename, "wb") as f:
            # write "magic" string
            f.write("P6\n".encode("utf-8"))
            # write size
            f.write((str(self.shape[2]) + " " + str(self.shape[1]) + "\n").encode("utf-8"))
            # write max value
            f.write("255\n".encode("utf-8"))

            # reshape to 1-d array
            reshaped = self.swapaxes(1, 2).flatten(order='F')
            
            # write image data
            reshaped.tofile(f)
            
        return self
