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

def f1(X) :
    
    r = np.sqrt(X[0]**2+X[1]**2)
    
    if X[1] < 0 : return(0)
    return(1+min(3, int(2*r)))
    
def f2(X) :
    
    r1 = np.sqrt(X[0]**2+X[1]**2)
    r2 = np.sqrt((X[0]-0.5)**2+X[1]**2)
    
    if (r1 > 1) or (r2 < 0.5) :
        return(1)
    else :
        return 0    
    
if __name__ == "__main__" :
        
      
    bounds = [[-2, -2], [2, 2]]
    
    clf = edsd.edsd(f1, X0=[[0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5]], bounds=bounds, processes=4, classes =5, N0 = 20, 
                    N1 = 1000, svc=dict(C = 100), neighbours = [[1,2], [2, 3], [3, 4], [0, 1], [0, 2], [0, 3], [0, 4]]) 
    
    
    clf.draw(grid_resolution = 1000, scatter=True)
    plt.show()

