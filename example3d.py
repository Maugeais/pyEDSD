#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import pyEDSD as edsd
import numpy as np
import matplotlib.pyplot as plt


def f1(X) :
    
    r = np.sqrt((X[0])**2+X[1]**2+(X[2])**2)

    return(r > 1)
    
if __name__ == "__main__" :
        
    
    bounds = [[-2, -2, -2], [2, 2, 2]]
    v = []
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    for N1 in range(100, 150, 100) :
        clf = edsd.edsd(f1, X0=[[0, 0, 0], [1, 1, 1]], bounds=bounds,  processes=4, 
                        N1 = N1, svc=dict(C = 1000), animate = False)
    
        v.append(4/3*np.pi-clf.volume(False))
    
    plt.semilogy(v)
    plt.grid(True)
    plt.xlabel("N1")
    plt.ylabel("Absolute error")
    
    plt.show()

