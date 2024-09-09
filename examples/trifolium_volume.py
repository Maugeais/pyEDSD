#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyEDSD as edsd
import numpy as np
import matplotlib.pyplot as plt     
import time      

if __name__ == "__main__" :
 
    clf =  edsd.load("trifolium.edsd")
    
    ax = clf.draw(options = [{"levels" : [0]}])

    clf.delaunay(class_id = 0, n_boundary = 200, n_interior = 2**10, draw = True, ax = ax)
    bb = clf.boundingbox(size_random =200)
    ax.set_xlim([bb[0][0]*1.1, bb[1][0]*1.1])
    ax.set_ylim([bb[0][1]*1.1, bb[1][1]*1.1])
    
    print(f"Volume of trifolium = {clf.volume(class_id = 0, n_boundary = 200, n_interior = 2**10)}")
    
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
