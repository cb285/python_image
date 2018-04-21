#!/usr/bin/env python3

import numpy as np
from math import cos, acos, sqrt, pow, radians
from copy import deepcopy
from subprocess import Popen
from os import setpgrp
import numbers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PADTYPE_NONE = "none"
PADTYPE_ZERO = "zero"
PADTYPE_SAME = "same"

IMSHOW_PROGRAM = "imagej"
#IMSHOW_PROGRAM = "emacs"

def imread(filename):
    
    # get file extension
    split_filename = filename.split(".")
    ext = split_filename[-1]
    
    # pbm
    if (ext == "pbm"):
        return read_pbm(filename)
        # pgm
    elif (ext == "pgm"):
        return read_pgm(filename)
        # ppm
    elif (ext == "ppm"):
        return read_ppm(filename)
        # png
    elif(ext == "png"):
        return read_png(filename)
    else:
        raise Exception("invalid/unsupported image format")
    
    return self

def read_png(filename):
    
    mimg = mpimg.imread(filename)
    if(len(mimg.shape) == 3):
        img = 255*mimg.transpose((2, 0, 1))
    else:
        img = 255 - 255*mimg
    img = img.astype(np.uint8)
    return img.view(Img)

def read_pbm(filename):
        
    with open(filename, "rb") as f:
        flines = f.read().split(bytes(b'\n'))
        
    magic = flines[0].decode("utf-8")
    n, m = map(int, flines[1].decode("utf-8").split(" "))
    dlines = flines[2:]
    
    img = np.empty((m, n), dtype=bool)
    
    # ASCII format
    if(magic == "P1"):
        
        data = list()

        #print(dlines)
        
        for line in dlines:
            line_words = line.decode("utf-8").split(" ")
            #print(line_words)
            for w in line_words:
                # remove special characters
                w = w.strip()
                # skip comments and empty words
                if(len(w) == 0):
                    continue
                elif(w[0] == "#"):
                    break
                data.append(w)

        # put data into array
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                img[j, i] = bool(int(data[img.shape[1]*j+i]))
                #print("data [", img.shape[1]*j+i, "] = " , 1 - int(data[img.shape[1]*j+i]))
                #print("img", (j,i), " = ", img[j,i])
        
    # binary format
    elif (magic == "P4"):
        
        data = dlines[0]
        
        pixel_num = 0
        byte_num = 0
        shift = 0
        
        # put data into array
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                pixel_num = img.shape[1]*j+i
                byte_num = int(pixel_num / 8)
                shift = 7 - pixel_num % 8
                    
                img[j, i] = bool((data[byte_num] & (1 << shift)) >> shift)

    else:
        raise Exception("invalid \"magic number\" for pbm")
    
    return img.view(Img)

def read_pgm(filename):
        f = open(filename, "rb")

        # read header
        magic = f.readline().decode("utf-8").strip()
        n, m = map(int, f.readline().decode("utf-8").strip().split(" "))
        max_val = int(f.readline().decode("utf-8").strip())

        img = np.empty((m, n), dtype=np.uint8)

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
            for j in range(img.shape[0]):
                for i in range(img.shape[1]):
                    img[j, i] = int(255*(float(data[img.shape[1]*j+i])/max_val))
        
        # binary format
        elif (magic == "P5"):
            # put data into array
            for j in range(img.shape[0]):
                for i in range(img.shape[1]):
                    img[j, i] = 255 - int.from_bytes(f.read(1), byteorder="big")

        else:
            raise Exception("invalid \"magic number\" for pbm")

        f.close()
        return img.view(Img)
    
def read_ppm(filename):
    f = open(filename, "rb")
    
    # read header
    magic = f.readline().decode("utf-8").strip()
    n, m = map(int, f.readline().decode("utf-8").strip().split(" "))
    max_val = int(f.readline().decode("utf-8").strip())
    
    img = np.empty((3, m, n), dtype=np.uint8)
    
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
        for j in range(img.shape[1]):
            for i in range(img.shape[2]):
                for p in range(3):
                    #print("img[",p,",",j,",",i,"] = ", int(255*(float(data[pixel_num])/max_val)))
                    img[p, j, i] = int(255*(float(data[pixel_num])/max_val))
                    pixel_num += 1

    # binary format
    elif (magic == "P6"):
        # put data into array
        for j in range(img.shape[1]):
            for i in range(img.shape[2]):
                for p in range(3):
                    img[p, j, i] = int(255*float(int.from_bytes(f.read(1), byteorder="big")) / max_val)

    else:
        raise Exception("invalid \"magic number\" for ppm")

    f.close()
    return img.view(Img)

class Img(np.ndarray):
    
    def __new__(subtype, shape=(1,1), dtype=np.uint8, buffer=None, offset=0,
                strides=None, order=None):
        
        obj = super(Img, subtype).__new__(subtype, shape, dtype,
                                          buffer, offset, strides,
                                          order)

        obj._padtype = None
        
        # fill array with zeros
        obj.fill(0)
        
        # return newly created instance
        return obj

    def __repr__(self):
        
        return np.array_str(self.view(type=np.ndarray))

    def __str__(self):

        return np.array_repr(self.view(type=np.ndarray))

    def __getitem__(self, index):

        #print(index)

        if(isinstance(index, slice)):
            #print(index)
            return super().__getitem__(index)
        
        # if index has one dimension
        if(isinstance(index, numbers.Number)):
            z = index

            # if within bounds
            if(z >= 0 and z < self.shape[0]):
                return super().__getitem__((z))

            # padding disabled
            if(self.padtype() == PADTYPE_NONE):
                raise IndexError("index " + str(index) + " out of bounds for image " + str(self.shape))
            # zero padding
            if(self.padtype() == PADTYPE_ZERO):
                return self.dtype(0)
            # same padding
            if(self.padtype() == PADTYPE_SAME):
                # if negative
                if(z < 0):
                    return self[0]
                # if too large
                else:
                    return super().__getitem__((self.shape[0] - 1))

        # if index has two dimensions
        if(len(index) == 2):

            # if image has less dimensions
            if(len(self.shape) < len(index)):
                raise IndexError("index " + str(index) + " out of bounds for image " + str(self.shape))
            
            y, x = index

            # if inside bounds
            if((y >= 0 and y < self.shape[0]) and (x >= 0 and x < self.shape[1])):
                return super().__getitem__((y, x))
            # if outside bounds
            else:
                # padding disabled
                if(self.padtype() == PADTYPE_NONE):
                    # raise error
                    raise IndexError("index " + str(index) + " out of bounds for image " + str(self.shape))
                # zero padding
                if(self.padtype() == PADTYPE_ZERO):
                    return 0
                # same padding

                if(self.padtype() == PADTYPE_SAME):
                    y, x = self._get_xy_same_ind(y, x)
                    return super().__getitem__((y, x))

        # if index has three dimensions
        if(len(index) == 3):

            # if image has less dimensions
            if(len(self.shape) < len(index)):
                raise IndexError("index " + str(index) + " out of bounds for image " + str(self.shape))

            z, y, x = index

            # if in bounds
            if((z >= 0 and z < self.shape[0]) and
               (y >= 0 and y < self.shape[1]) and
               (x >= 0 and x < self.shape[2])):
                return super().__getitem__((z, y, x))
            
            y, x = self._get_xy_same_ind(y, x, 1)
            
            if(z < 0):
                return self[0, y, x]
            elif(z >= self.shape[0]):
                return super().__getitem__((self.shape[0] - 1, y, x))
            else:
                return self[z, y, x]

        else:
            raise Exception("more than three dimension indexing is not supported")

    def _get_xy_same_ind(self, y, x, y_index=0):

        #print("index =", y, x)
        
        if(y < 0):
            y = 0

        elif(y >= self.shape[y_index]):
            y = self.shape[y_index] - 1

        if (x < 0):
            x = 0

        elif(x >= self.shape[y_index+1]):
            x = self.shape[y_index+1] - 1

        return y, x

    def padtype(self, padtype=False):

        if(not hasattr(self, "_padtype")):
            self._padtype = PADTYPE_NONE
        
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
        if(padtype != False):
            # save original padtype
            orig_padtype = self.padtype()
            # change padtype
            self.padtype(padtype)
        else:
            if(self.padtype() == PADTYPE_NONE):
                raise ValueError("padtype cannot be none")
            orig_padtype = self.padtype()
            
        # create copy of image
        original_image = self.copy()
        original_image.padtype(self.padtype())
        
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
                        s += mask[mj, mi] * original_image[j - mask_mid + mj, i - mask_mid + mi]
                #print(s)
                self[j, i] = s

        # restore pad type
        self._padtype = orig_padtype
                
        return self

    def imshow(self, figure=1, block=True):

        if(len(self.shape) == 3):
            show_img = self.transpose((1, 2, 0)) / 255
        else:
            show_img = self

        plt.figure(figure)
        plt.imshow(show_img, cmap='Greys')
        plt.show(block=block)
        
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

    def _write_png(self, filename):

        if(len(self.shape) == 3):
            write_img = self.transpose((1, 2, 0)) / 255
            mpimg.imsave(filename, write_img)
        else:
            write_img = self
            mpimg.imsave(filename, write_img, cmap='Greys')
    
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
