#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pyEDSD as edsd

def trifolium(X) :
    a = 1
    res =  (X[0]**2+X[1]**2)*(X[1]**2+X[0]*(X[0]+a))-4*a*X[0]*X[1]**2
    
    return(res > 0)


           

if __name__ == "__main__" :
 
    clf =  edsd.edsd(trifolium, X0=[[-0.5, 0], [0.25, 0.25], [0.25, -0.25]], bounds= [[-2, -2], [2, 2]], 
                            processes=10, classes = 2, verbose = True,
                            N0 = 100, N1 = 100, svc=dict(C = 1000))

    

    clf = clf.set_random_box([[-0.3, -0.3], [0.3, 0.3]])

    clf = clf.expand(1000, processes=10, verbose = True)

    clf = clf.set_random_box([[-2, -2], [2, 2]])

    ax = clf.draw(scatter = True, options = [{"levels" : [0]}], scatter_options={'marker':'x'})
    
    clf.show()

   