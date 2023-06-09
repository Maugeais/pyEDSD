#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import edsd
import numpy as np


def f1(X) :
    
    r = np.sqrt(X[0]**2+X[1]**2+X[2]**2)
    
    if r > 0.5 : 
        return 1
    else :
        return 0
  
bounds = [[-0.8, -0.8, -0.8], [0.8, 0.8, 0.8]]
clf = edsd.edsd(f1, X0=[[0, 0, 0], [0.5, 0.5, 0.5]], bounds=bounds, N1 = 1000, C = 100)  
h = .1  # step size in the mesh

clf.draw(h)

