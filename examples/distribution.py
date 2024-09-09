#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:57:46 2023

@author: maugeais
"""

import pyEDSD as edsd
import numpy as np
import matplotlib.pyplot as plt
import time, scipy

def circle(X) :
    r = np.sqrt(X[0]**2+X[1]**2)
    
    return(r > 1)
    
def paramcircle(t) :

    return(np.array([np.cos(t), np.sin(t)]).T)

def offCenteredCircle(X) :
    r = np.sqrt((X[0]-1)**2+(X[1]-1)**2)
    
    return(r > 1)
    
def paramoffCenteredCircle(t) :

    return(np.array([np.cos(t)+1, np.sin(t)+1]).T)

def ellipsis(X) :
    a, b = 1, 0.5
    r = np.sqrt(X[0]**2/a**2+X[1]**2/b**2)
    
    if r > 1 : 
        return 1
    else :
        return 0
    
def paramellipsis(t) :
    a, b = 1, 0.5

    return(np.array([a*np.cos(t), b*np.sin(t)]).T)

def lemniscate(X) :
    
    a = 1
    res = (X[0]**2+X[1]**2)**2-a*(X[0]**2-X[1]**2)
    
    return(res > 0)

def paramlemniscate(t) :
    a = 1

    return(np.array([a*np.sin(t)/(1+np.cos(t)**2), a*np.sin(t)*np.cos(t)/(1+np.cos(t)**2)]).T)


def trifolium(X) :
    a = 1
    res =  (X[0]**2+X[1]**2)*(X[1]**2+X[0]*(X[0]+a))-4*a*X[0]*X[1]**2
    
    return(res > 0)

def paramtrifolium(t) :
    a = 1
    r = a*np.cos(t)*(4*np.sin(t)**2-1)
    
    return(np.array([r*np.cos(t), r*np.sin(t)]).T)


def canonicalParam(func, T, N) :
    """ Computation of curvilinear abscissa
    
    """
    t = np.arange(0, T+2*T/N, T/N)
    F = globals()['param'+func.__name__](t)
    
    normedF = np.linalg.norm(F[2:, :]-F[:-2, :], axis=1)/(t[1]-t[0])/2
    
    s = np.cumsum(1/normedF)*(t[1]-t[0])
 
    return(s, F)    

def findClosest(X, F) :
    """ find the closest point to X in F, returns the index """
    
    I = np.argsort(np.linalg.norm(F-np.array(X), axis = 1))[0]
                    
    return(I)

def distribution(X, F) :
    """ Compute the distribution of F """
    
    dist = np.zeros(F.shape[0])
    indeces = []
        
    for x in X :
        
        I = findClosest(x, F)
        indeces.append(I)
        
        dist[I] += 1
        
    
    return(np.cumsum(dist)/len(X), indeces)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def inversion(y, z):

    x = np.zeros_like(y)
    N = len(y)

    for i in range(N) :

        j = 0
        while (y[j] <= z[i] and j < N-1):
            j+=1

        x[i] = j/N

    return(x)


def compare_to_ref(func, N1 = 100, size = 1000, curvature = False, label = "", clf = None, ref = []) :

    
    # Computes the parametrisation
    s1, F1 = canonicalParam(func, 2*np.pi, 10000)

    
    # Plotting of the distribution law against curvature
    fig=plt.figure("Distribution") #Figure(figsize=(7, 3.5))
    axis = plt.gca() #fig.add_subplot(1, 1, 1)
    print(f"Generation with {label}")

    # Resets the random pool
    clf.reset_random_pool()

    try :
        X = clf.random(size=size, processes = 10)
    except :
        print(f"Problem with the generation of random points using the generator {label}")
        return()

    dist, I = distribution(X, F1[:-2])
    rand = s1[I]/max(s1)

    if len(ref) == 0 :
        return(dist)
 
    
    s1 = np.interp(np.linspace(0, 1, len(ref)), np.linspace(0, 1, len(s1)), s1)
    dist = np.interp(np.linspace(0, 1, len(ref)), np.linspace(0, 1, len(dist)), dist)
    discrepancy = inversion(dist, ref)
    discr = np.reshape(discrepancy[1:]-discrepancy[:1], (len(discrepancy)-1, 1))
    
    discr = np.reshape(dist[1:]-dist[:1], (len(dist)-1, 1))
    
    try : 
        from scipy.stats import qmc
        label = f"discrepancy -> {qmc.discrepancy(discr):.5f}"
    except :
        print("The library 'scipy.stats.qmc' is not available")
        label=""

    axis.plot(s1, dist, label=label)
    plt.xlabel("Parametrisation")
    plt.ylabel("Empirical distribution")
        
        
    plt.grid(True)
    plt.legend()
    plt.tight_layout()



if __name__ == "__main__" :
      
    # Test on trifolium
    clf =  edsd.load("trifolium.edsd")

    ref = []
    ref = compare_to_ref(trifolium, size=10000, label =  "Reference", clf= clf)
    size = 100

    compare_to_ref(trifolium, size=size, label =  "numpy", clf= clf, ref = ref)


    plt.show()
