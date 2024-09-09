#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyEDSD as edsd
import numpy as np
import matplotlib.pyplot as plt


def generate_random(size = 1000, clf = None, ax = None) :
    
    
    X = clf.random(size=size, processes = 1, verbose = True)
    X = np.array(X)
    print(min(abs(clf.decision_function((X-clf._b)/clf._a))))
    

    ax.scatter(X[:, 0], X[:, 1])
        
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
            

if __name__ == "__main__" :
 
    clf =  edsd.load("trifolium.edsd")
    
    ax = clf.draw(options = [{"levels" : [0]}])

    generate_random(size=100, clf= clf, ax = ax)

    plt.show()
