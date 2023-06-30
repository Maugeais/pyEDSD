#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:14:00 2023

@author: maugeais

Roughly  translates http://codes.arizona.edu/Research/EDSD#bas08a to python
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from scipy.optimize import minimize
from scipy.spatial.distance import pdist

from sklearn.inspection import DecisionBoundaryDisplay
import multiprocessing
import time
import sys


from skimage import measure

colors = ['r', 'g', 'b', 'c', 'm', 'k']

class svcEDSD(svm.SVC):
    
    def draw(self, grid_resolution = 100):
        
        if (len(self.bounds[0]) == 2) : 
            self.draw2d(grid_resolution = grid_resolution)
            
        elif (len(self.bounds[0]) == 3) : 
            self.draw3d(grid_resolution = grid_resolution)


    def draw2d(self, grid_resolution = 100, scatter = True) :
            
        X = self.trainingSet
        y = self.predict(X)
 
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            self,
            self.trainingSet,
            ax=ax,
            grid_resolution=grid_resolution,
            plot_method="contourf",
            colors=colors,
            levels=len(self.classes_)-2,
            alpha=0.5,
            # linestyles=["--"]*len(self.classes_),
        )
        ax.set_aspect(1)
        plt.xlim(self.bounds[0][0], self.bounds[1][0])
        plt.ylim(self.bounds[0][1], self.bounds[1][1])
        if scatter :
            
            for i, c in enumerate(self.classes_) :
                
                I = np.where(y == c)[0]
                plt.scatter(X[I, 0], X[I, 1], c=colors[i], cmap=plt.cm.coolwarm, marker = 'x', alpha =0.5, label="class "+str(c)) 
                
        plt.legend()
        
    def draw3d(self, grid_resolution = 10, scatter = True) :
            
        X = self.trainingSet
        y = self.predict(X)
        bounds = self.bounds
    
        
        # create a mesh to plot in
        x_min, x_max = bounds[0][0], bounds[1][0]
        h = (x_max-x_min)/grid_resolution
        y_min, y_max = bounds[0][1], bounds[1][1]
        z_min, z_max = bounds[0][2], bounds[1][2]
        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h),
                             np.arange(z_min, z_max, h))
        
        
        
        F = self.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()]).reshape(xx.shape)
            
        verts, faces, normals, values = measure.marching_cubes(F, 0, spacing=[h, h, h])
        verts -= bounds[1]
        
        
        ax = plt.gca()
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, lw=0, antialiased=True)
        
        if scatter : 
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
        plt.tight_layout()
        
    def _random(self, id=0) :
        
        np.random.seed((2**3*id+time.time_ns())%(2**32))
            
        if len(self.classes_) > 2 :
                                            
            if self.decision_function_shape == 'ovo' :
                    
                classNumber = np.random.randint((len(self.classes_))*(len(self.classes_)-1)//2)
                
            else :
                
                classNumber = np.random.randint(len(self.classes_))
                                    
            decision = lambda x : self.decision_function([x])[0][classNumber]**2 #+1/(1+dist(x, X)**2)
                
        else :
            
            decision = lambda x : self.decision_function([x])**2 #+1/(1+dist(x, X)**2)
            
        while True : 
            
            x0 = self.bounds[0]+(self.bounds[1]-self.bounds[0])*np.random.rand(len(self.bounds[0]))
            
            res = minimize(decision, x0, method='CG')#, bounds=bounds)
            
            
            # Si on est dans les bounds
            if all([(res.x[i] > self.bounds[0][i]) and (res.x[i] < self.bounds[1][i]) for i in range(len(self.bounds[0]))]) :
                
                return(res.x)

    def resetRandomPool(self) :
        
        self._randomPool = []
        
    def random(self, size=1, processes = 1) :
 
        if size == 1 : 
            
            return(self._random())
      
        if not hasattr(self, '_randomPool') :
            
            self._randomPool = []
            
        newSize = max(size-len(self._randomPool), 0)
               
        with multiprocessing.Pool(processes=processes) as pool:
        
            for r in pool.map(partial(_parallel_, rand=self._random), range(newSize)):
                    
                self._randomPool.append(r)
                
        I = np.random.randint(0, len(self._randomPool), size=size)
                
        return([self._randomPool[i] for i in I])

    def diameter_estimate(self, size_random = None) :
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
        
        X = self.random(size=size_random)
        
        return(max(pdist(X)))
    
    def boundingbox_estimate(self, size_random = None) :
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
    
        X = np.array(self.random(size=size_random))
        
        return([np.array([min(X[:, 0]), min(X[:, 1])]), np.array([max(X[:, 0]), max(X[:, 1])])])
    
    def distFrom(self, P=[], size_random=None) :
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
    
        X = np.array(self.random(size=size_random))
        
        plt.scatter(*P)
        return(min(np.linalg.norm(X-np.array(P), axis=1)))
    
    def volume(self, size_random=None) :
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
            
        bounds = self.bounds #self.boundingbox_estimate()
    
        X = bounds[0]+(bounds[1]-bounds[0])*np.random.rand(size_random, len(self.bounds[0]))
        
        y = self.predict(X)
        
        V = np.prod([bounds[1][i]-bounds[0][i] for i in range(len(bounds[0]))])
        
        return (V*(1-y.sum()/len(y)))


from functools import partial 
def _parallel_(id = 0, rand=None, func=None) :
                
    x = rand(id = id)
    if func != None :
        y = func(x)
        return([x, y])
    
    else :
        return(x)

# Il faut au moins un point dans chaque classe    
def edsd(func, X0=[], bounds=[], N0 = 10, N1 = 10, processes = 1, classes = 1, 
         verbose = True, animate = False,
         svc={}) :
    """ Explicit Design space decomposition
    
    Parameters :
    ----------
        func : function(n-dimensional vector) -> 0 or 1, function to classify
        X0 : list of n-dimension vectors
            
                must contain at leat one point in each class
        bounds : list of 2 n-dimensionnal vector
        
        N0 : int, optional (default = 10)
        
            number of points to be added to the training set before initialising the svm
            chosen random uniformly between bounds to initialise the svm
            
        N1 : int, optional (default = 10)
        
            number of points to be added to the training set, 
            chosen near the decision function 
            
        processes : int, optional (default = 1)
            
             if processes > 1, then the commputation is paralellized, and the SVC is updated
            only after "processes" points on the boundary have been found
            
        params* : all the parameters of an SVC (cf . sklearn.svm.SVC)
    
    Returns : 
    ----------
        an svcEDSD, which is an svc to which are added 
        
        1 methods
        
            draw : function(h)
                    displayes the decision bounday together with the training sets
                    
        2 objects            
        
    References
    ----------
    .. [1] Basudhar, A., and Missoum, S., 
           “Adaptive explicit decision functions for probabilistic 
           design and optimization using support vector machines”, 
           Computers & Structures, vol. 86, 2008, pp. 1904 - 1917.

    """

    X = X0.copy()
    y =[]
    
    processes = min(multiprocessing.cpu_count(), processes)
     
    bounds = [np.array(x) for x in bounds]    
   
    # Ajout de points aléatoires uniform dans bounds
    
    if verbose :
        print("Creation of first set")
        
    for n in range(N0) :
        
        x = bounds[0]+(bounds[1]-bounds[0])*np.random.rand(len(bounds[0]))
        X.append(list(x))
        
        
    # Calcul des valeurs des fonctions
    n = 0        
    with multiprocessing.Pool(processes=processes) as pool:
                
        for r in pool.map(func, X):
            
            y.append(r)
            
            if verbose : 
                n += 1
                sys.stdout.write('\r')
                # the exact output you're looking for:
                i = int(100*n/len(X))
                sys.stdout.write("[%-20s] %d%%" % ('='*(i//5), i))
                sys.stdout.flush()   
                
        
    if classes > 2 :
        svc["decision_function_shape"]='ovo'
        
        
    clf = svm.SVC(**svc).fit(X, y)
    clf.bounds = bounds
    clf.__class__ = svcEDSD
    
    
    if verbose : 
        print("\nClasses found : ", clf.classes_)

    if len(bounds[0]) > 3 :
        animate = False       
    
    n = 0
    
    ax = plt.gca()
    while n < N1//processes :
        
        if processes == 1 :
            
            x0 = clf._random()
            X.append(x0)
            y.append(func(x0))
        
        else : 
            
            # result = clf.random(N=processes)
            # X.extend(result)
            result = []
            
            with multiprocessing.Pool(processes=processes) as pool:
                
                for r in pool.map(partial(_parallel_, rand=clf._random, func=func), range(processes)):
                      
                    X.append(r[0])
                    y.append(r[1])
                
        clf = svm.SVC(**svc).fit(X, y)
        clf.bounds = bounds
        clf.__class__ = svcEDSD
        clf.trainingSet = np.array(X)
        
        if animate :
            
            clf.draw()

            plt.savefig('/tmp/img'+format(n, '05d')+'.jpg', dpi=200)
            ax.clear()
            
        if verbose : 
            sys.stdout.write('\r')
            # the exact output you're looking for:
            i = int(100*n*processes/N1)
            sys.stdout.write("[%-20s] %d%%" % ('='*(i//5), i))
            sys.stdout.flush()
        
        n += 1
    
    if verbose : 
        print("\nFinal set of classes : ", clf.classes_)
    
            
    if animate :
        import os 
        os.system("ffmpeg -r 10 -i /tmp/img%05d.jpg -vcodec mpeg4 -y movie.mp4")       
        os.system("rm /tmp/img*.jpg")       
              
    return(clf)
    
    
