#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import pyEDSD as edsd
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.spatial as spatial


def circle(X) :
    
    r = np.sqrt(X[0]**2+X[1]**2)
    
    return(r > 1)
    
def f2(X) :
    
    r1 = np.sqrt(X[0]**2+X[1]**2)
    r2 = np.sqrt((X[0]-0.5)**2+X[1]**2)
    
    if (r1 > 1) or (r2 < 0.5) :
        return(1)
    else :
        return 0    
    
def lemniscate(X) :
    
    a = 1
    res = (X[0]**2+X[1]**2)**2-a*(X[0]**2-X[1]**2)
    
    return(res > 0) 

def trifolium(X) :
    a = 1
    res =  (X[0]**2+X[1]**2)*(X[1]**2+X[0]*(X[0]+a))-4*a*X[0]*X[1]**2
    
    return(res > 0)    


    
    
if __name__ == "__main__" :

    bounds = [[-2, -2], [2, 2]]
    
    t0 = time.time()

    clf = edsd.edsd(circle, X0=[[-0.5, 0], [0.5, 0], [1, 1]], bounds=bounds, processes=4, classes = 2, verbose = True,
                    N1 = 100, svc=dict(C = 1000, gamma = 1))

    
    print("EDSD computation time", time.time()-t0)


    ax = clf.draw()

    clf.delaunay(class_id = 0, n_boundary = 200, n_interior = 2**10, draw = True, ax = ax)
    
    t0 = time.time()

    print(clf.volume(class_id = 0, n_boundary = 200, n_interior = 2**10))

    print("Volume computation time", time.time()-t0)

    plt.show()