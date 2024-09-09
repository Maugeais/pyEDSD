#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pyEDSD as edsd

def trifolium(X) :
    a = 1
    res =  (X[0]**2+X[1]**2)*(X[1]**2+X[0]*(X[0]+a))-4*a*X[0]*X[1]**2
    
    return(res > 0)


           

if __name__ == "__main__" :
 
    # edsd.random.set_random_generator("Sobol")
    clf =  edsd.edsd(trifolium, X0=[[-0.5, 0], [0.25, 0.25], [0.25, -0.25]], bounds= [[-2, -2], [2, 2]], 
                            processes=10, classes = 2, verbose = True,
                            N0 = 1000, N1 = 3000, svc=dict(C = 1000))

    edsd.save(clf, "trifolium.edsd")
   