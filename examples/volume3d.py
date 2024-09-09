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
    clf = edsd.edsd(f1, X0=[[0, 0, 0], [1, 1, 1]], bounds=bounds,  processes=20, 
                        N1 = 2000, svc=dict(C = 1000, gamma=1))

    n_boundary = 10
    n_interior = 500

    for n_interior in range(300, 2000, 200) : # This is the most efficient variable to compute the volume
    # for n_boundary in range(100, 600, 100) :
        
        # clf.reset_random_pool() # Choose to reset or not
    
        v.append(4/3*np.pi-clf.volume(class_id = 0, n_boundary = n_boundary, n_interior = n_interior))
    
    plt.semilogy(v)
    plt.grid(True)
    plt.xlabel("n_interior")
    plt.ylabel("Absolute error")
    plt.tight_layout()
    plt.show()
