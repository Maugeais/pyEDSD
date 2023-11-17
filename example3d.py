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
    
    r = np.sqrt((X[0]-0.5*X[2])**2+X[1]**2+0.3*(X[2]-0.1*X[1])**2)
    
    if r > 0.5 : 
        return 1
    else :
        return 0
    
if __name__ == "__main__" :
        
    
    bounds = [[-2, -2, -2], [2, 2, 2]]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    clf = edsd.edsd(f1, X0=[[0, 0, 0], [0.5, 0.5, 0.5]], bounds=bounds,  processes=4, 
                    N1 = 100, svc=dict(C = 100), animate = False)
    
    clf.draw()

    #clf.contour3d(scatter=False)
    
    plt.show()

