#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:45:18 2023

@author: maugeais
"""

from scipy.stats import qmc

class np_random(qmc.QMCEngine):

    def __init__(self, d, seed=None):

        super().__init__(d=d, seed=seed)



    def _random(self, n=1, *, workers=1):

        print("hola")
        return self.rng.random((n, self.d))



    def reset(self):

        super().__init__(d=self.d, seed=self.rng_seed)

        return self



    def fast_forward(self, n):

        self.random(n)

        return self

    def random(self, n) :
        return(np.random.rand(n, self.d))

import numpy as np
import warnings
import sys, os

sampler = None
dimension = 0
initialiser = qmc.LatinHypercube

def set_random_generator(name = "Sobol") :

    global initialiser
    if name == "numpy" :
        initialiser = np_random

    if name == "Sobol" :
        initialiser = qmc.Sobol

    if name == "Halton" :
        initialiser = qmc.Halton

    if name == "Latin" :
        initialiser = qmc.LatinHypercube

    if name == "Poisson" :
        initialiser = qmc.PoissonDisk
 

def rand(d, n = 1, seed=0) :

    global sampler, initialiser
    if dimension != d or sampler == None:
        
        sampler = initialiser(d=d)

    res = sampler.random(n=n)

    if n == 1 :
        return(res[0])

    return(res) #.random(n=1)[0])
    

def discrepency(sample) :

    return(qmc.discrepancy(sample))
