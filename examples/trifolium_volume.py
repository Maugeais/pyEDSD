#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyEDSD as edsd
import numpy as np
import matplotlib.pyplot as plt           

if __name__ == "__main__" :
 
    
    edsd.random.set_random_generator("Sobol")
    clf =  edsd.load("trifolium.edsd")
    
    ax = clf.draw(options = [{"levels" : [0]}])

    clf.delaunay([0, 1], value = -1, n_boundary = 200, n_interior = 2**10, draw = True, ax = ax)

    ax.set_xlim([-1.15, 0.75])
    ax.set_ylim([-1, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()
