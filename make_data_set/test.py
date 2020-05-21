# -*- coding: utf-8 -*-
"""
Created on Mon May 18 22:29:37 2020

@author: Administrator
"""


import os
import h5py
import numpy as np

def create_h5():
    # shape = 4392  2x16x8
    imgData = np.zeros((4392,2,16,8))

    print(imgData,"====",len(imgData))

    if not os.path.exists('1.h5'):
        with h5py.File('1.h5') as f:
            f['data'] = imgData
            f['labels'] = range(100)
            
def read_h5():
    with h5py.File('test.hdf5') as f:
        print(f)
        print(f.keys)

if __name__ == "__main__":
    create_h5()
            