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

    if r < 1  :
        return(0)
    # if r < 2 :
    #     return(1)
    # if r < 3 :
    #     return(2)
    return(3)
    
if __name__ == "__main__" :
        
    
    bounds = [[-4, -4, -4], [4, 4, 4]]
    v = []
    
    edsd.set_backend("plotly")

    
    clf = edsd.edsd(f1, X0=[[0, 0, 0], [1, 1, 1]], bounds=bounds,  processes=4, N0 = 10,
                        N1 = 200, svc=dict(C = 1000), neighbours=[[0, 1], [1, 2], [2, 3]])

    edsd.save(clf, "3d.edsd")

