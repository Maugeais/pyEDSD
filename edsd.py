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


from skimage import measure


class svcEDSD(svm.SVC):
    
    def draw(self, h):
        
        if (len(self.bounds[0]) == 2) : 
            self.draw2d(h)
            
        elif (len(self.bounds[0]) == 3) : 
            self.draw3d(h)


    def draw2d(self, h, scatter = True) :
            
        X = self.trainingSet
        y = self.predict(X)
        bounds = self.bounds
        
        # create a mesh to plot in
        # x_min, x_max = bounds[0][0], bounds[1][0]
        # y_min, y_max = bounds[0][1], bounds[1][1]
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                      np.arange(y_min, y_max, h))
    
    
        # Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # # Put the result into a color plot
        # Z = Z.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            self,
            self.trainingSet,
            ax=ax,
            grid_resolution=100,
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
        plt.show()
        
    def draw3d(self, h, scatter = True) :
            
        X = self.trainingSet
        y = self.predict(X)
        bounds = self.bounds
    
        
        # create a mesh to plot in
        x_min, x_max = bounds[0][0], bounds[1][0]
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
        plt.show()    
        
        
    def random(self, N) :
        n = 0 
        
        X=  []
        
        while n < N :
        
        
            x0 = self.bounds[0]+(self.bounds[1]-self.bounds[0])*np.random.rand(len(self.bounds[0]))
            
            decision = lambda x : self.decision_function([x])**2 #+1/(1+dist(x, X)**2)
                    
            res = minimize(decision, x0, method='CG')#, bounds=bounds)
            
            
            # Si on est dans les bounds
            if all([(res.x[i] > self.bounds[0][i]) and (res.x[i] < self.bounds[1][i]) for i in range(len(self.bounds[0]))]) :
                X.append(list(res.x))
            
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
def edsd(func, X0=[], bounds=[], N0 = 10, N1 = 10, animate = False,
         C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None) :
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
    
    
    for x in X :
        y.append(func(x))
        
    bounds = [np.array(x) for x in bounds]    
    
    # Ajout de points aléatoire uniform dans bounds
    for n in range(N0) :
        
        x = bounds[0]+(bounds[1]-bounds[0])*np.random.rand(len(bounds[0]))
        X.append(list(x))
        y.append(func(x))
        
        
    clf = svm.SVC(kernel=kernel, gamma='scale', C=C).fit(X, y)
    
    if len(bounds[0] != 2) :
        animate = False
    if animate :
        ax = plt.gca()
        plt.xlim(bounds[0][0], bounds[1][0])
        plt.ylim(bounds[0][1], bounds[1][1])
    # Recherche de points sur les zéros
    
    n = 0
    while n < N1 :
        x0 = bounds[0]+(bounds[1]-bounds[0])*np.random.rand(len(bounds[0]))
        
        decision = lambda x : clf.decision_function([x])**2 #+1/(1+dist(x, X)**2)
                
        res = minimize(decision, x0, method='CG')#, bounds=bounds)
        
        
        # Si on est dans les bounds
        if all([(res.x[i] > bounds[0][i]) and (res.x[i] < bounds[1][i]) for i in range(len(bounds[0]))]) :
            
            if animate :
                ax.set_aspect(1)
                plt.xlim(bounds[0][0], bounds[1][0])
                plt.ylim(bounds[0][1], bounds[1][1])
                
                DecisionBoundaryDisplay.from_estimator(
                    clf,
                    np.array(X),
                    ax=ax,
                    grid_resolution=100,
                    plot_method="contour",
                    colors="k",
                    levels=[-1, 0, 1],
                    alpha=0.5,
                    linestyles=["--", "-", "--"],
                )
                plt.savefig('/tmp/img'+format(n, '05d')+'.jpg', dpi=100)
                ax.clear()
            
            X.append(list(res.x))
            y.append(func(res.x))
            clf = svm.SVC(kernel='rbf', gamma='scale', C=C).fit(X, y)
            
            n += 1 
            
    if animate :
        import os 
        os.system("ffmpeg -r 10 -i /tmp/img%05d.jpg -vcodec mpeg4 -y movie.mp4")       
        os.system("rm /tmp/img*.jpg")       
            
    ### Add methods and data to class            
                
    clf.trainingSet = np.array(X)
    clf.bounds = bounds
    
    clf.__class__ = svcEDSD

   
    
    return(clf)
    
    
