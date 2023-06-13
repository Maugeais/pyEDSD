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
    
    if r > 1 : 
        return 2
    elif r > 0.5 :
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

clf = edsd.edsd(f1, X0=[[0, 0], [0.5, 0.5], [1, 1]], bounds=bounds, processes=3, classes =3, 
                N1 = 500, svc=dict(C = 1000), animate = False) 


plt.figure()
clf.draw()
plt.show()

