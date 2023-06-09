#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import edsd
import numpy as np
import matplotlib.pyplot as plt
import time

def f1(X) :
    
    r = np.sqrt(X[0]**2+X[1]**2)
    
    if r > 0.5 : 
        return 1
    else :
        return 0
    
def f2(X) :
    
    r1 = np.sqrt(X[0]**2+X[1]**2)
    r2 = np.sqrt((X[0]-0.5)**2+X[1]**2)
    
    if (r1 > 1) or (r2 < 0.5) :
        return(1)
    else :
        return 0    
  
bounds = [[-2, -2], [2, 2]]
clf = edsd.edsd(f2, X0=[[-0.5, 0], [0.5, 0], [1, 1]], bounds=bounds, N1 = 1000, C = 1000, animate = True) 

h = .005  # step size in the mesh

plt.figure()
clf.draw(h)

