#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from subprocess import Popen
from os import setpgrp

class ImageReadError(Exception):
    pass

class Img():
    def __init__(self, arg=(0,0)):
        
        # dimensions
        if(type(arg) is tuple):
            self._size = arg
            self._img_data = np.zeros(self._size, np.uint8)
            self._dtype = "uint8"
            self._rep = "gray"
            
        # filename
        elif(type(arg) is str):
            self.imread(arg)
        
        # image
        elif(type(arg) is type(self)):
            self._size = deepcopy(arg._size) # tuples are immutable, but their elements might not be
            self._img_data = arg._img_data.copy() # arrays are mutable
            self._dtype = arg._dtype # strings are immutable, so no copy needed
            self._rep = arg._rep

    def __repr__(self):
        s = "size:" + str(self._size[0]) + "x" + str(self._size[1]) + "\n"
        s = s + "dtype:" + self._dtype + "\n"
        s = s + "rep:" + self._rep + "\n"
        s = s + str(self._img_data)
        return s
    
    def __getitem__(self, tup):
        if(len(tup) == 3):
            y, x, z = tup
            return self._img_data[z, y, x]
        
        if(len(tup) == 2):
            y, x = tup
            if (len(self._size) == 3):
                return self._img_data[:, y, x]
            else:
                return self._img_data[y,x]
            
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
            raise ImageReadError("invalid/unsupported image format")
            #self._write_pgm(filename)
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
            
            l = f.read().split(bytes(b'\n'))
            print(str(l[0]))
            #lines = f.readlines()
        
        words = list()
        
        for line in lines:
            line_words = line.split(" ")
            for w in line_words:
                w.strip()
                # skip comments
                if(w[0] == "#"):
                    break
                # skip empty words
                if(w[0] == ""):
                    continue
                words.append(w.strip())
        
        words = [w for w in words if w]
                
        # get "magic number"
        magic = words[0]
        # get size
        size = (int(words[2]), int(words[1])) # (m, n)
        # get data
        data = words[3:]
        
        # ASCII format
        if (magic == "P1"):
            
            # store image info
            self._size = size
            self._img_data = np.zeros(self._size, np.uint8)
            self._dtype = "uint8"
            self._rep = "bw"
            
            # put data into array
            for j in range(self._size[0]):
                for i in range(self._size[1]):
                    self._img_data[j, i] = int(data[self._size[1]*j+i])
                    
        # binary format
        elif (magic == "P4"):
            pass
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
    
    def _read_pgm(self, filename):
        raise ImageReadError("unsupported image format")
    
    def _read_ppm(self, filename):
        raise ImageReadError("unsupported image format")
