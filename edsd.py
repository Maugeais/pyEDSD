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


from skimage import measure


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
            plot_method="contour",
            colors="k",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
        )
        ax.set_aspect(1)
        plt.xlim(self.bounds[0][0], self.bounds[1][0])
        plt.ylim(self.bounds[0][1], self.bounds[1][1])
        if scatter :
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, marker = 'x', alpha =0.5) 
        
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
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, lw=0, antialiased=True)
        
        if scatter : 
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
        plt.tight_layout()
        
        
    def random(self, id=0, N=1) :
        n = 0 
                
        X=  []
                
        np.random.seed((2**3*id+time.time_ns())%2**32)
        
        while n < N :
        
        
            x0 = self.bounds[0]+(self.bounds[1]-self.bounds[0])*np.random.rand(len(self.bounds[0]))
            
            decision = lambda x : self.decision_function([x])**2 #+1/(1+dist(x, X)**2)
                    
            res = minimize(decision, x0, method='CG')#, bounds=bounds)
            
            
            # Si on est dans les bounds
            if all([(res.x[i] > self.bounds[0][i]) and (res.x[i] < self.bounds[1][i]) for i in range(len(self.bounds[0]))]) :
                X.append(list(res.x))
                
                if N == 1 :
                    return(res.x)
            
                n+=1
        
        return(X)  

    def diameter_estimate(self, N = None) :
        
        if N == None :
            N = 10**len(self.bounds[0])
        
        X = self.random(N)
        
        return(max(pdist(X)))
    
    def boundingbox_estimate(self, N = None) :
        
        if N == None :
            N = 10**len(self.bounds[0])
    
        X = np.array(self.random(N))
        
        return([np.array([min(X[:, 0]), min(X[:, 1])]), np.array([max(X[:, 0]), max(X[:, 1])])])
    
    def volume(self, N=None) :
        
        if N == None :
            N = 10**len(self.bounds[0])
            
        bounds = self.bounds #self.boundingbox_estimate()
    
        X = bounds[0]+(bounds[1]-bounds[0])*np.random.rand(N, len(self.bounds[0]))
        
        y = self.predict(X)
        
        V = np.prod([bounds[1][i]-bounds[0][i] for i in range(len(bounds[0]))])
        
        return (V*(1-y.sum()/len(y)))

    
def dist(x, X) :
    
    d = min([np.linalg.norm(x-z) for z in X])**2
    
    return d

    

# Il faut au moins un point dans chaque classe    
def edsd(func, X0=[], bounds=[], N0 = 10, N1 = 10, processes = 1, animate = False,
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
    
    for x in X :
        y.append(func(x))
        
    bounds = [np.array(x) for x in bounds]    
    
    # Ajout de points aléatoires uniform dans bounds
    for n in range(N0) :
        
        x = bounds[0]+(bounds[1]-bounds[0])*np.random.rand(len(bounds[0]))
        X.append(list(x))
        y.append(func(x))
        
        
    clf = svm.SVC(**svc).fit(X, y)
    clf.bounds = bounds
    clf.__class__ = svcEDSD

    if len(bounds[0]) > 3 :
        animate = False       
    
    n = 0
    
    ax = plt.gca()
    while n < N1//processes :
        
        if processes == 1 :
            
            x0 = clf.random()
            X.append(x0)
            y.append(func(x0))
        
        else : 
            with multiprocessing.Pool(processes) as pool:
                       
                for result in pool.map(clf.random, range(processes)):
                    
                    X.append(result)
                    y.append(func(result))
              
                
        clf = svm.SVC(**svc).fit(X, y)
        clf.bounds = bounds
        clf.__class__ = svcEDSD
        clf.trainingSet = np.array(X)

        
        if animate :
            
            clf.draw()

            plt.savefig('/tmp/img'+format(n, '05d')+'.jpg', dpi=100)
            ax.clear()
            
        n += 1
            
    if animate :
        import os 
        os.system("ffmpeg -r 10 -i /tmp/img%05d.jpg -vcodec mpeg4 -y movie.mp4")       
        os.system("rm /tmp/img*.jpg")       
              
    return(clf)
    
    
