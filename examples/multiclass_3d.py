#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pyEDSD as edsd
import numpy as np


def f1(X) :
    
    r1 = ((X[0]-2)**2+X[1]**2+X[2]**2)
    r2 = (X[0]**2+(X[1]-2)**2+X[2]**2)
    r3 = (X[0]**2+X[1]**2+(X[2]-2)**2)

    if r1 < 1  :
        return("0")
    if r2 < 1 :
        return("1")
    if r3 < 1 :
        return("2")
    return("3")
    
if __name__ == "__main__" :
        

    bounds = [[-4, -4, -4], [4, 4, 4]]
    v = []
    
    
    clf = edsd.edsd(f1, X0=[[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 1]], bounds=bounds,  processes=4, classes = 4, N0 = 100,
                        N1 = 400, svc=dict(C = 1000), neighbours=[])
    
    edsd.save(clf, "multi_3d.edsd")

