#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import pyEDSD as edsd
import numpy as np


def f1(X) :
    
    r = np.sqrt((X[0]+2)**2+X[1]**2+(X[2]-1+0.5*X[1])**2)

    if r < 1  :
        return(0)
    if r < 2 :
        return(1)
    if r < 3 :
        return(2)
    return(3)
    return(r > 1)
    
if __name__ == "__main__" :
        
    
    bounds = [[-4, -4, -4], [4, 4, 4]]
    v = []
    
    
    clf = edsd.edsd(f1, X0=[[-2, 0, 1], [1, 1, 1]], bounds=bounds,  processes=4, classes = 4, 
                        N1 = 300, svc=dict(C = 1000))
    
    clf.draw(classes = [1, 2], scatter = True, mayavi = True)

    clf.show()

