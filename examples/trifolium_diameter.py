#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyEDSD as edsd
import numpy as np
import matplotlib.pyplot as plt     
import time      

def computation_time(processes) :
    
    diameters = []
    clf.reset_random_pool()
    times = []
    N = range(50, 1500, 50)
    for n in N :
        t0 = time.time()
        clf.reset_random_pool()
        diameters.append(clf.diameter_estimate(class_id = 0, size_random = n, processes = processes))
        times.append(time.time()-t0)
        
    return(N, diameters, times)
    
    

if __name__ == "__main__" :
 
    
    clf =  edsd.load("trifolium.edsd")
    

    v0 = clf.diameter_estimate(class_id = 0, size_random = 2000, processes = 20)
   
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    N, vol, times = computation_time(1)
    ax1.scatter(N, abs(1-vol/v0), c='b')
    ax2.scatter(N, times, c='c', marker="P", label="1 worker")
    N, vol, times = computation_time(20)
    ax1.scatter(N, abs(1-vol/v0), c='r')
    ax2.scatter(N, times, c='m', marker="P", label="20 worker")
    ax1.set_yscale('log')

    ax1.set_xlabel("Number of points on the boundary")
    ax1.set_ylabel("Relative error in the estimation of diameter")
    ax2.set_ylabel("Computation time (s)")
    
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()