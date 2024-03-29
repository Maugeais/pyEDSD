#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import pyEDSD as edsd
import numpy as np
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
    
    return ((r1 > 1) or (r2 < 0.5))
        

if __name__ == "__main__" :

    bounds = [[-2, -2], [2, 2]]
    
    t0 = time.time()

    clf = edsd.edsd(f2, X0=[[-0.5, 0], [0.5, 0], [1, 1]], bounds=bounds, processes=20, classes = 2, verbose = True,
                    N0 = 20, N1 = 500, svc=dict(C = 1000))
    
   
    print("Temps de calcul", time.time()-t0)

    clf.draw(scatter = True)
    
    # print(clf.dist_from([0.5, 0]))
    
    clf.show()
