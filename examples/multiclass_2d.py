#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import pyEDSD as edsd
import numpy as np


def func(X) :
    
    r = np.sqrt(X[0]**2+X[1]**2)
    
    if X[1] < 0 : return(0)
    return(1+min(3, int(2*r)))
    
    
if __name__ == "__main__" :
        
      
    bounds = [[-2, -2], [2, 2]]
    
    clf = edsd.edsd(func, X0=[[0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5]], bounds=bounds, processes=4, classes =5, N0 = 20, 
                    N1 = 1000, svc=dict(C = 100), neighbours = [[1,2], [2, 3], [3, 4], [0, 1], [0, 2], [0, 3], [0, 4]]) 
    
    edsd.save(clf, "multi.edsd")

