#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:57:46 2023

@author: maugeais
"""

import edsd
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


def testFunction(func, N1 = 100, size = 1000, curvature = False, changeBounds = False) :

    bounds = [[-2, -2], [2, 2]]
    
    # Computation of edsd
    plt.figure('SVM')

    clf = edsd.edsd(func, X0=[[-0.5, 0], [0.25, 0.25], [0.25, -0.25]], bounds=bounds, processes=4, classes = 2, verbose = True,
                    N0 = 100, N1 = N1, svc=dict(C = 100), animate = False)
    
    clf.draw()
    
    # Drawing of the parametrisation
    s1, F1 = canonicalParam(func, 2*np.pi, 10000)

    plt.plot(F1[:, 0], F1[:, 1])
    
    # Plotting of the probability law against curvature
    fig=plt.figure("Proba law") #Figure(figsize=(7, 3.5))
    axis = plt.gca() #fig.add_subplot(1, 1, 1)
    
    
    if changeBounds :
        clf.chgRandomBounds(False)
    
    
    X = clf.random(size=size, processes = 4, verbose = True)
    
    dist, I = distribution(X, F1[:-2])
    rand = s1[I]/max(s1)
    
    distUnif = s1/max(s1)
    axis.plot(s1, distUnif, label="Uniform distribution")
    
    axis.plot(s1, dist, label="edsd distribution")
    plt.xlabel("Arc length t")
    plt.ylabel("F(t)")
        
        
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if curvature :
        axisp = axis.twinx()
        R = np.linalg.norm((F1[2:-1]-2*F1[1:-2]+F1[:-3]), axis=1)/(s1[1:]-s1[:-1])**2
        axisp.plot(s1[1:],  R, c='r', alpha=  0.5)

    
    # Statistic test against uniform law
    # print("Kolmogorov Smirnov test", scipy.stats.kstest(rand, 'uniform', args=(min(rand), max(rand))))
    
            

if __name__ == "__main__" :
    
    # Test on the centered circle
    # testFunction(circle, size=10000, changeBounds = False)
    # testFunction(circle, size=10000, changeBounds = True)
    
    # Test on the off centered circle
    # testFunction(offCenteredCircle, size=10000, changeBounds = False)
    # testFunction(offCenteredCircle, size=10000, changeBounds = True)
    
    # Test on trifolium
    testFunction(ellipsis, N1 = 1000, size=10000, changeBounds = False)
    testFunction(ellipsis, N1 = 1000, size=10000, curvature=True, changeBounds = True)

    plt.show()