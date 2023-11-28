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
import pickle


import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.insert(0, dir_path)


from skimage import measure
import tools
from functools import partial 
import scipy.spatial as spatial




class svcEDSD(svm.SVC):
    
    def draw(self, grid_resolution = 100, scatter = False, fig = None):
        """
        Draw the zones and their boundaries obtained by the classifier
        Parameters
        ----------
        grid_resolution : int, optional
            DESCRIPTION. resolution of the grid used to draw the contour.
            The default is 100.
        scatter : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """        
        if fig == None :
            fig = plt.figure()

        if (len(self.bounds[0]) == 2) : 
            self._draw2d(grid_resolution = grid_resolution, scatter = scatter, fig = fig)
            
        elif (len(self.bounds[0]) == 3) : 
            self._draw3d(grid_resolution = grid_resolution, scatter = scatter, fig = fig)

        else :
            print("Cannot draw in mod than 3d")


    def _draw2d(self, grid_resolution = 100, scatter = True, fig = None) :
        """
        Draw the zones and their boundaries obtained for a 2d classifier
        Parameters
        ----------
        grid_resolution : int, optional
            DESCRIPTION. resolution of the grid used to draw the contour.
            The default is 100.
        scatter : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        X = self.trainingSet
        y = self.predict(X)
 
        ax = plt.gca()
        if len(self.classes_) == 2 :
            levels = 0
        else :
            levels = len(self.classes_)-1
        DecisionBoundaryDisplay.from_estimator(
            self,
            self.trainingSet,
            ax=ax,
            grid_resolution=grid_resolution,
            plot_method="contourf",
            levels=levels,
            colors=tools.colors,
            alpha=0.5,
        )
        # ax.set_aspect(1)
        plt.xlim(self.bounds[0][0], self.bounds[1][0])
        plt.ylim(self.bounds[0][1], self.bounds[1][1])
        if scatter :
            
            for i, c in enumerate(self.classes_) :
                
                I = np.where(y == c)[0]
                plt.scatter(X[I, 0], X[I, 1], c=tools.colors[i], marker = 'x', alpha =0.5, label="class "+str(c)) 
                
            plt.legend()
        
    def contour3d(self, grid_resolution = 10, scatter = True, fig = None) :
        """
        Draw the contour of a 3d classifier

        Parameters
        ----------
        grid_resolution : int, optional
            DESCRIPTION. resolution of the grid used to draw the contour.
            The default is 100.
        scatter : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.


        """

        if fig == None :
            fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
            
        X = self.trainingSet
        y = self.predict(X)
        bounds = self.bounds
    
        
        # create a mesh to plot in
        x_min, x_max = bounds[0][0], bounds[1][0]
        y_min, y_max = bounds[0][1], bounds[1][1]
        z_min, z_max = bounds[0][2], bounds[1][2]
        
        def f(x, y, z) :
            
            if type(x) != np.ndarray :
                
                Y = y.ravel()
                Z = z.ravel()
                X = x + 0*Z
                ref = y.shape
                
            if type(y) != np.ndarray :
                
                X = x.ravel()
                Z = z.ravel()
                Y = y + 0*Z
                ref = x.shape
                
            if type(z) != np.ndarray :
                
                Y = y.ravel()
                X = x.ravel()
                Z = z + 0*X
                ref = x.shape
                
                
            return(self.decision_function(np.c_[X, Y, Z]).reshape(ref))
        
        tools.plot_implicit(ax, f, bbox=bounds)
        
        
        if scatter : 
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
        plt.tight_layout()
        
    def _draw3d(self, grid_resolution = 10, scatter = True, fig = None) :
        """Draw the zones and their boundaries obtained for a 3d the classifier
        Parameters
        ----------
        grid_resolution : int, optional
            DESCRIPTION. resolution of the grid used to draw the contour.
            The default is 100.
        scatter : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        ax = fig.add_subplot(111, projection='3d')

            
        X = self.trainingSet
        y = self.predict(X)
        bounds = self.bounds
    
        
        # create a mesh to plot in
        x_min, x_max = bounds[0][0], bounds[1][0]
        hx = (x_max-x_min)/grid_resolution
        y_min, y_max = bounds[0][1], bounds[1][1]
        hy = (y_max-y_min)/grid_resolution
        z_min, z_max = bounds[0][2], bounds[1][2]
        hz = (z_max-z_min)/grid_resolution
        xx, yy, zz = np.mgrid[x_min:x_max:hx, y_min:y_max:hy, z_min:z_max:hz]
    
        
        F = self.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()]).reshape(xx.shape)
            
        verts, faces, normals, values = measure.marching_cubes(F, 0, spacing=[hx, hy, hz])
        verts -= bounds[1]
        
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, lw=0, antialiased=True)
        
        if scatter : 
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
        plt.tight_layout()

    def show(self) :
        plt.show()
        
        
    def change_random_bounds(self, X, error = 0.05) :
        """
        Change the bounds for the random operations
        
            if X is a set of coordinates then
            if X is a class then
            
            the error term is used ....        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        error : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        None.

        """            
        if not hasattr(self, '_randBounds') :
            
            self._randBounds = self.bounds.copy()
            
        if X in self.classes_ :
            
            X = self.classBounds[X]
            
        d = [X[1][j]-X[0][j] for j in range(len(X))]
            
        self._randBounds = [np.array([max(self.bounds[0][j], X[0][j]-error*d[j]) for j in range(len(X))]),
                                     np.array([min(self.bounds[1][j], X[1][j]+error*d[j]) for j in range(len(X))])]
                    
        
    def _random(self, id=0) :
        """
        Main random function, it should not be called directly

        Parameters
        ----------
        id : int, optional
            parameter used to initialise the random seed, together with the clock.
            This is present to prevent many calls to _random sent at the same time 
            to give the same result, which can happen if the clock resolution used 
            for the initialization of the seed is too coarse.
            The default is 0.

        Raises
        ------
        Exception 'random'
            If a new point has not been found within the boundaries after a 100 trials

        Returns
        -------
        A point close to the one of the boundaries of the classifier

        """
                
        np.random.seed((2**3*id+time.time_ns())%(2**32))
        
        if not hasattr(self, '_randBounds') :
            
            self._randBounds = self.bounds.copy()
            
        if len(self.classes_) > 2 :
                                            
            if self.decision_function_shape == 'ovo' :
                    
                classNumber = np.random.randint((len(self.classes_))*(len(self.classes_)-1)//2)
                
            else :
                
                classNumber = np.random.randint(len(self.classes_))
                                    
            decision = lambda x : self.decision_function([x])[0][classNumber]**2 #+1/(1+dist(x, X)**2)
                
        else :
            
            decision = lambda x : self.decision_function([x])**2 #+1/(1+dist(x, X)**2)
            
        n = 0
                        
        while True : 
            
            x0 = self._randBounds[0]+(self._randBounds[1]-self._randBounds[0])*np.random.rand(len(self._randBounds[0]))
                        
            res = minimize(decision, x0, method='CG')#, bounds=bounds)
             
            # If the result is within the bounds
            if all([(res.x[i] > self._randBounds[0][i]) and (res.x[i] < self._randBounds[1][i]) for i in range(len(self._randBounds[0]))]) :
                
                return(res.x)
            
            n += 1
            
            if n > 100 :
                print("Error in teh generation of the random points, try ...")
                raise Exception('random')

    def reset_random_pool(self) :
        """
        Reset the random pool

        Returns
        -------
        None.

        """
        
        self._random_pool = []
        
    def random(self, size=1, processes = 1, verbose = False) :
        """
        Fetch size elements in the random pool. 
        It the pool does not have enough precomputed elements, 
        then it uses the paralellization of the self._random function
        to fill it

        Parameters
        ----------
        size : int, optional
            number of points. 
            The default is 1.
        processes : int, optional
            number of processes. 
            The default is 1.
        verbose : boolean, optional
            If True, an advanscement bar is displayed in the console.
            The default is False.

        Returns
        -------
        None.

        """
         
        if size == 1 : 
            
            return(self._random())
      
        if not hasattr(self, '_random_pool') :
            
            self._random_pool = []
            
        new_size = max(size-len(self._random_pool), 0)
        
        n = 0
         
        with multiprocessing.Pool(processes=processes) as pool:
                    
            for r in pool.map(partial(tools._parallel, func1=self._random), range(new_size)):
                    
                self._random_pool.append(r)

                if verbose : 
                    n += 1
                    tools._advBar(int(100*n/new_size))  
                    
        if verbose :
            print("")
                
        I = np.random.randint(0, len(self._random_pool), size=size)
                
        return([self._random_pool[i] for i in I])
    
    def diameter_estimate(self, size_random = None) :
        """
        Estimates the diameter of the boundary using a Monte Carlo method.
        The points are taken from the random pool.

        Parameters
        ----------
        size_random : int, optional
            Number of random points to get in the random pool. 
            If None, the number of random points is taken to be 10**d 
            where d is the dimension of the search space
            The default is None.

        Returns
        -------
        None.

        """
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
        
        X = self.random(size=size_random)
        
        return(max(pdist(X)))
    
    def boundingbox_estimate(self, size_random = None) :
        """
        Estimates the minimal bounding box for the boundaries of the classifier 
        using a Monte Carlo method.
        The points are taken from the random pool.

        Parameters
        ----------
        size_random : int, optional
            Number of random points to get in the random pool. 
            If None, the number of random points is taken to be 10**d 
            where d is the dimension of the search space
            The default is None.

        Returns
        -------
        None.

        """
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
    
        X = np.array(self.random(size=size_random))
        
        return([np.array([min(X[:, 0]), min(X[:, 1])]), np.array([max(X[:, 0]), max(X[:, 1])])])
    
    def dist_from(self, P=[], size_random=None) :
        """
        Estimates the distance from a point to the boundary using a Monte Carlo method.
        The points are taken from the random pool.

        Parameters
        ----------
        size_random : int, optional
            Number of random points to get in the random pool. 
            If None, the number of random points is taken to be 10**d 
            where d is the dimension of the search space
            The default is None.

        Returns
        -------
        None.


        """
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
    
        X = np.array(self.random(size=size_random))
        
        plt.scatter(*P)
        return(min(np.linalg.norm(X-np.array(P), axis=1)))
    
    def volume(self, value, n_boundary = 0, n_interior = 10) :
        """
        Compute the volume of one region given by the classifier defined by value.
        For this, a delaunay triangulation is computing using :
            - the points of the training set
            - points on the boundary, using the random function
            - points in the interior, obtained using n_interior**dim points 
                chosen randomly within the boundaries, and sorted using the classifier

        Parameters
        ----------
        value : TYPE
            DESCRIPTION.
        n_boundary : TYPE, optional
            DESCRIPTION. The default is 0.
        n_interior : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        """
    
        # Compute a Dalaunay triangulation
        
        # points = clf.trainingSet[np.logical_not(clf.trainingSetValues)]
        points = self.trainingSet[self.trainingSetValues == value]
        val = self.decision_function([points[0]])
        
        dim = len(self.bounds[0])
        

        # Add points in the interior
        if n_interior > 0 :
            rand = np.random.uniform(min(self.bounds[0]), max(self.bounds[1]), dim*n_interior**dim).reshape(n_interior**dim, dim)
            I = (self.decision_function(rand)*val > 0)
            
            points = np.concatenate((points, rand[I]))
        
        # Add points on the boundary
        if n_boundary > 0 :
            rand = self.random(n_boundary)
            points = np.concatenate((points, rand))
        
        tri = spatial.Delaunay(points)
                
        # Remove simplicies whose centre is outside 
        
        I = [simp for simp in list(tri.simplices) if self.decision_function([np.mean(tri.points[simp], axis = 0)])*val > 0]

        tri.simplices = I    
        setattr(spatial._qhull.Delaunay, 'volume', tools.volume)
        
        # try : 
        #     # fig = plt.figure()
        #     # ax = fig.add_subplot(111, projection='3d')
        #     plt.scatter(tri.points[:, 0], tri.points[:, 1], tri.points[:, 2])
        #     # spatial.delaunay_plot_2d(tri)
        # plt.triplot(points[:,0], points[:,1], tri.simplices)
    
        #     # # plt.plot(points[:,0], points[:,1], 'o')
    
        #     # plt.show()
        # except :
        #     pass
        return(tri.volume())
    
    def expand(self, N1, processes = 4, verbose = False, animate = False) :
        """
        Create a new classifier from the current one by adding N1 new points

        Parameters
        ----------
        N1 : TYPE
            DESCRIPTION.
        processes : TYPE, optional
            DESCRIPTION. The default is 4.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.
        animate : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        n = 0
        X = self.trainingSet.tolist()
        y = self.trainingSetValues.tolist()
        clf = self
                        
        while n < N1//processes :
                     
            if processes == 1 :
                
                x0 = clf._random()
                X.append(x0)
                y.append(self.func(x0))
            
            else : 
                
                try :
                
                    with multiprocessing.Pool(processes=processes) as pool:
                        
                        for r in pool.map(partial(tools._parallel, func1=clf._random, func2=self.func), range(processes)):
                            
                            if r[1] != -1 :
                                X.append(r[0])
                                y.append(r[1])
                except KeyboardInterrupt:
                    print('\nStopping at {}%'.format(100*n*processes/N1))
                    clf = _fit(self.func, self.svc, self.bounds, X, y)
                    break
                except BaseException as error:
                    print(error)
                    clf = _fit(self.func, self.svc, self.bounds, X, y)
                    
                    break
    
            clf = _fit(self.func, self.svc, self.bounds, X, y)        
                
            if verbose : 
                tools._advBar(int(100*n*processes/N1))
            
            n += 1
            
        # # Get the bounds for all the classes in a dictionnary
        clf.classBounds = dict()
             
        for c in clf.classes_ :
             
            I = np.where(y == c)[0]
            clf.classBounds[c] = [np.array([min([X[i][j] for i in I]) for j in range(len(X[0]))]), 
                                      np.array([max([X[i][j] for i in I]) for j in range(len(X[0]))])]  
                                        
        return(clf)

    
def _fit(func, svc, bounds, X, y) :
    """
    

    Parameters
    ----------
    svc : TYPE
        DESCRIPTION.
    bounds : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    clf = svm.SVC(**svc).fit(X, y)
    clf.bounds = bounds
    clf.__class__ = svcEDSD
    clf.trainingSet = np.array(X)
    clf.trainingSetValues = np.array(y)
    clf.func = func
    clf.svc = svc    

    return(clf)
    
def save(clf, filename) :
    """
    Save the classifier to a pickle file

    Parameters
    ----------
    clf : svcEDSD
        DESCRIPTION.
    filename : string
        DESCRIPTION.

    Returns
    -------
    None.

    """
    func = clf.func
    clf.func = None
    with open(filename, 'wb') as file:
        pickle.dump(clf, file) 
    clf.func = func


def load(filename) :
    """
    Load a classifier

    Parameters
    ----------
    filename : string
        DESCRIPTION.

    Returns
    -------
    svcEDSD

    """
    with open(filename, 'rb') as file:
        res = pickle.load(file)     
    return(res)
    
def edsd(func, X0=[], bounds=[], N0 = 10, N1 = 10, processes = 1, classes = 1, 
         verbose = True, svc={}) :
    """ Explicit Design space decomposition
    
    Parameters :
    ----------
        func : function(n-dimensional vector) -> 0 or 1, function to classify, if return of f is -1, then not a class
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
                tools._advBar(int(100*n/len(X)))

    # Remove the elements in the "error" class
    I = [i for i in range(len(y)) if y[i] != -1]
    if len(I) > 0 and verbose :
        print("\nRemoving {} elements for class -1, stored in sef.removed".format(len(X)-len(I)))
    
    removed = [X[i] for i in range(len(y)) if y[i] != -1]   

    X = [X[i] for i in I]
    y = [y[i] for i in I]
        
    if classes > 2 :
        svc["decision_function_shape"]='ovo'
        
        
    clf = _fit(func, svc, bounds, X, y)
    
    if verbose : 
        print("Classes found : ", clf.classes_)
        print("Number of points in each class :", [str(c) + ':' + str(len(np.where(y==c)[0])) for c in clf.classes_])
            
    if len(bounds[0]) > 3 :
        animate = False       
    
    clf = clf.expand(N1, processes = processes, verbose = verbose)

    clf.removed = removed

    # Get the bounds for all the classes in a dictionnary
    clf.classBounds = dict()
        
    for c in clf.classes_ :
        
        I = np.where(y == c)[0]
                
        clf.classBounds[c] = [np.array([min([X[i][j] for i in I]) for j in range(len(X[0]))]), 
                                np.array([max([X[i][j] for i in I]) for j in range(len(X[0]))])]  
                                      
    if verbose : 
        print("\nFinal set of classes : ", clf.classes_)
              
    return(clf)
    
    
