#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import pyEDSD as edsd
import numpy as np


def f1(X) :
    
    r = np.sqrt((X[0])**2+X[1]**2+(X[2])**2)

    return(r > 1)
    
if __name__ == "__main__" :
        
    
    bounds = [[-2, -2, -2], [2, 2, 2]]
    v = []
    
    
    clf = edsd.edsd(f1, X0=[[0, 0, 0], [1, 1, 1]], bounds=bounds,  processes=4, 
                        N1 = 100, svc=dict(C = 1000))
    
    clf.draw(scatter = True)

    clf.show()

