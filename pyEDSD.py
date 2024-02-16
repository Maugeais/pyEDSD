#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:14:00 2023

@author: maugeais

Roughly  translates http://codes.arizona.edu/Research/EDSD#bas08a to python
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from sklearn import svm

from scipy.optimize import minimize
from scipy.spatial.distance import pdist

import multiprocessing
import time
import pickle

from lib import plot, tools, random

__version__ = "0.4.1"

from functools import partial 
import scipy.spatial as spatial


    
plot._backend = "matplotlib"

def set_backend(backend = "matplotlib") :
    
    if backend in ["matplotlib", "mayavi", "plotly"] :
        plot._backend = backend
    else :
        print("The backend you asked for is not supported")
        
        

max_random_gradient = 100 # Maximal number of unconclusive gradient iteration

class svcEDSD(svm.SVC):
    
    def draw(self, plot_method = "frontiers", grid_resolution = 100, scatter = False, 
             contour = False, classes = [], frontiers = [], ax = None, fig = None, options = [{}], label_options = [{}], scatter_options = [{}]):
        """
        Draw the zones and their boundaries obtained by the classifier
        Parameters
        ----------
        plot_method : {'frontiers', 'classes', 'mesh'}, default is contour, mesh is only for 3d
        grid_resolution : int, optional
            DESCRIPTION. resolution of the grid used to draw the contour.
            The default is 100.
        scatter : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """        

        if (self.dimension_ == 2) : 
            if plot_method == 'frontiers' : 
                return(plot._frontiers2d(self, grid_resolution = grid_resolution, scatter = scatter, frontiers = frontiers, ax = ax, 
                                       options = options, label_options = label_options, scatter_options = scatter_options))
            
            elif plot_method == "classes" :
                return(plot._classes2d(self, grid_resolution = grid_resolution, scatter = scatter, classes = classes, ax = ax, 
                            options = options, label_options = label_options, scatter_options = scatter_options))
            
            
            else :
                raise ValueError(f"the value {plot_method} is not possible for the vairable plot_method")
                
            
        elif (self.dimension_ == 3) : 

            if plot_method == 'mesh' :
                plot._contour3d(self, grid_resolution = grid_resolution, scatter = scatter, classes = classes, 
                                ax = ax, options = options, scatter_options = scatter_options)
            elif plot_method == 'frontiers' :
                return(plot._frontiers3d(self, grid_resolution = grid_resolution, scatter = scatter, frontiers = frontiers, 
                                         ax = ax, fig = fig, options = options, scatter_options = scatter_options))
            elif plot_method == 'classes' :
                return(plot._classes3d(self, grid_resolution = grid_resolution, scatter = scatter, classes = classes, 
                                       ax = ax, fig = fig, options = options, scatter_options = scatter_options))

        else :
            print("Cannot draw in more than 3d")
            
            

    def show(self, fig = None) :

        plot.show(fig)
        
    def index_neighbourhood_(self, n) :
        
        i0 = np.where(self.classes_ == n[0])[0][0]
        i1 = np.where(self.classes_ == n[1])[0][0]
        i0, i1 = min(i0, i1), max(i0, i1)
                     
        m = 0
        for j in range(i0) :
            m += len(self.classes_)-j-1
            
        m += i1-i0-1
              
        return(m)
    
    def restriction(self, indices, values) :
        
        clf = svcEDSD()
        
        I = [i for i in range(self.dimension_) if i not in indices]
        # print(I)
        
        clf._a, clf._b = self._a[I], self._b[I]
        clf.trainingSet = []
        clf.trainingSetValues = []
        clf.classes_ = self.classes_
        clf.decision_function_indices_ = self.decision_function_indices_
        clf.neighbours_ = self.neighbours_
        clf.dimension_ = self.dimension_-len(indices)
        clf.decision_function = clf.restricted_decision_function
        clf.parent = self
        clf.indices_ = indices
        clf.values_ = (np.array(values)-self._b[indices])/self._a[indices]
        
        return(clf)
        
    def restricted_decision_function(self, x) :
        
        x = np.array(x)
        xp = np.zeros((x.shape[0], x.shape[1]+len(self.indices_)))
        
        k = 0
        for i in range(x.shape[1]+len(self.indices_)) :
            if i in self.indices_ :
                j = self.indices_.index(i)
                xp[:, i] = self.values_[j]
            else :
                xp[:, i] = x[:, k]
                k += 1
                
        y = self.parent.decision_function(xp)
        # print(y)
        return(y)


    def choose_multi_decision_function_(self, class_id = -1) :
        
        if class_id < 0 :
            classNumber = np.random.choice(self.decision_function_indices_)
            decision = lambda x : self.decision_function([x])[0][classNumber]**2 #+1/(1+min(np.linalg.norm(self.trainingSet-x, axis=1)**2))
        
        else :
                                
            decision = lambda x : self.class_decision_function([x], class_id=class_id)**2
            
        return(decision)
        
    def random_(self, id=0, class_id = -1) :
        """
        Main random function, it should not be called directly

        Parameters
        ----------
        id : int, optional
            parameter used to initialise the random seed, together with the clock.
            This is present to prevent many calls to random_ sent at the same time 
            that would  give the same result, which can happen if the clock resolution 
            used for the initialization of the seed is too coarse.
            The default is 0.

        Raises
        ------
        Exception 'random'
            If a new point has not been found within the boundaries after a 100 trials

        Returns
        -------
        A point close to the one of the boundaries of the classifier

        """
        tol = 1e-3
        
        # np.random.seed((2**3*id+time.time_ns())%(2**32))
            
        if len(self.classes_) > 2 :
            
            decision = self.choose_multi_decision_function_(class_id = class_id) 
                
        else :
            
            decision = lambda x : self.decision_function([x])**2 #+1/(1+dist(x, X)**2)
            
        n = m = 0

        # Try first the official pool
        x0 = random.get_from_pool(id)
        
        while True : 
                 
            res = minimize(decision, x0, method='CG', options = {})#, bounds=bounds)
            
            # If the result is within the bounds
            if  res.fun < tol and all([(res.x[i] > 0) and (res.x[i] < 1) for i in range(self.dimension_)]) :
                return(self._a*res.x+self._b)         
            
            if n > max_random_gradient :
                                               
                if len(self.classes_) == 2 or class_id >= 0 :
                    raise Exception('random')
                    
                m += 1
                
                if m > len(self.classes_) :
                    raise Exception('random')              
                                
                decision = self.choose_multi_decision_function_(class_id = class_id) 

                n = 0

            # if it did not work with original pool, try another random element
            n += 1
            x0 = random.rand(self.dimension_)     

    def reset_random_pool(self, class_id = 0) :
        """
        Reset the random pool of the given class_id
        
        Parameters
        ----------
        class_id : optional
            id of the class 

        Returns
        -------
        None.

        """

        # if not hasattr(self, 'random_pool') :
            
        #     self.random_pool = []
        
        self.random_pool_ = []
        
    def random(self, size=1, processes = 1, verbose = False, class_id = -1) :
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
        class_id : optional
            id of the class 

        Returns
        -------
        None.

        """
         
        
      
        if not hasattr(self, 'random_pool_') :
            
            self.random_pool_ = []
               
        new_size = max(size-len(self.random_pool_), 0)

        random.generate_pool(d = self.dimension_, n = new_size)
        
        if size == 1 :     
            return(self.random_())
        
        n = 0
         
        with multiprocessing.Pool(processes=processes) as pool:
                    
            # for r in pool.map(self.random_, range(new_size)):
            for r in pool.map(partial(parallel_random_, func = self.random_, class_id = class_id), range(new_size)) :
                    
                self.random_pool_.append(r)

                if verbose : 
                    n += 1
                    tools._advBar(int(100*n/new_size))  
                    
        if verbose :
            print("")
                
        I = np.random.randint(0, len(self.random_pool_), size=size)
                
        return(np.array([self.random_pool_[i] for i in I]))
    
    def diameter_estimate(self, size_random = None, class_id = -1, processes = 1) :
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
            size_random = 10**self.dimension_
        
        X = self.random(size=size_random, class_id = class_id, processes = processes)
        
        return(max(pdist(X)))
    
    def boundingbox(self, size_random = None, class_id = -1, processes = 1) :
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
            size_random = 10**self.dimension_
            
        try :
    
            X = np.array(self.random(size=size_random, class_id = class_id, processes = processes))
        
    
            return([np.array([min(X[:, 0]), min(X[:, 1])]), np.array([max(X[:, 0]), max(X[:, 1])])])
    
        except :
            return([[0, 0], [0, 0]])
    
    def dist_from(self, P=[], size_random=None, class_id = -1, processes = 1) :
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
            size_random = 10**self.dimension_
    
        X = np.array(self.random(size=size_random, class_id = class_id, processes = processes))
        
        plot.scatter(*P)
        return(min(np.linalg.norm(X-np.array(P), axis=1)))
    
    def delaunay(self, class_id = -1, frontier = [0, 1], n_boundary = 0, n_interior = 10, draw = False, 
                 ax = None, processes = 1) :
        """
        Compute a delaunay triangulation one region given by the classifier .
        For this, a delaunay triangulation is computing using :
            - the points of the training set
            - points on the boundary, using the random function
            - points in the interior, obtained using n_interior**dim points 
                chosen randomly within the boundaries, and sorted using the classifier
                
        As Delaunay triangulation depends on the metric, it is computed with inhomogenous data

        Parameters
        ----------
        class_id : TYPE
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
        # frontier = [min(frontier), max(frontier)]
        # if len(self.classes_) > 2 :
        #     index = self.decision_function_indices_[self.neighbours_.index(frontier)]
        # else :
        #     index = 0
      
        # Creates the points in the boundary
        # points = self.random(n_boundary, class_id = index)
        points = self.random(n_boundary, class_id = class_id, processes = processes)

        m, M = np.min(points, axis = 0), np.max(points, axis = 0)
        
        # Then the points in the interior
        n_interior = max(n_interior, 2**self.dimension_)

        random.generate_pool(self.dimension_, n_interior)
        

        rand = (M-m)*random.pool_+m
        
        

        if len(self.classes_) == 2 :
                        
            decision = self.homogenous_decision_function
            
        else :
        
            decision = lambda x : self.homogenous_class_decision_function(x, class_id=class_id)
            
        I = decision(rand) < 0           
        
        points = np.concatenate((points, rand[I, :]))

        tri = spatial.Delaunay((points-self._b)/self._a)
        tri.points[:] = self._a*tri.points[:] + self._b
        
                
        tri.simplices = [simp for simp in list(tri.simplices) if decision([np.mean(tri.points[simp], axis = 0)]) < 0]
        
        if draw :
            
            if self.dimension_ == 2 :
                        
                ax.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices)
     
        return(tri)
    
        
    
    def volume(self, class_id, n_boundary = 0, n_interior = 10, processes = 1) :
        """
        Compute the volume of one region given by the classifier defined by value.

        Parameters
        ----------
        class_id : TYPE
            DESCRIPTION.
        n_boundary : TYPE, optional
            DESCRIPTION. The default is 0.
        n_interior : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.
        """
        
        tri = self.delaunay(class_id, n_boundary, n_interior, processes = processes)
        
        setattr(spatial._qhull.Delaunay, 'volume', tools.volume)
        
        return(tri.volume())

    def homogenous_decision_function(self, X) :
        return(self.decision_function((X-self._b)/self._a))
    
    def class_decision_function(self, x, class_id) :
        res = np.max([(-float(class_id==n[0])+float(class_id==n[1]))*self.decision_function(x)[:, self.index_neighbourhood_(n)] for n in self.neighbours_ if class_id in n], axis = 0)
        return(res)
    
    def homogenous_class_decision_function(self, X, class_id) :
        return(self.class_decision_function((X-self._b)/self._a, class_id=class_id))
       
    
    def expand(self, N1, processes = 4, verbose = False, animate = False,  neighbours = []) :
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

        if self.func == None :
            print("The original function does not exist")
            return()
        n = 0
        X = self.trainingSet.tolist()
        y = self.trainingSetValues.tolist()
        clf = self
        
        while n < N1//processes :
                
            if processes == 1 :
                
                x0 = clf.random_()
                X.append(x0)
                y.append(self.func(x0))
            
            else :         
 
                try :
                
                    with multiprocessing.Pool(processes=processes) as pool:
                        
                        for r in pool.map(partial(parallel_random_eval_, func1=clf.random_, func2=self.func), range(processes)):

                            if r[1] != -1 :
                                X.append(r[0])
                                y.append(r[1])
                                
                except KeyboardInterrupt:
                    print('\nStopping at {}%'.format(100*n*processes/N1))
                    clf = fit_(self.func, self.svc, X, y, self._a, self._b, neighbours)
                    break
                
                except BaseException as error:
                    print(error)
                    clf = fit_(self.func, self.svc, X, y, self._a, self._b, neighbours)
                    
                    break
    
            clf = fit_(self.func, self.svc, X, y, self._a, self._b, neighbours)    
                
            if verbose : 
                tools._advBar(int(100*n*processes/N1))
            
            n += 1
                                        
        return(clf)

def relevant_distances(classes, neighbours) :
    """
    Compute the indeces of the distance for the set of neighbouring regieons stored in neighbours

    Parameters
    ----------
    classes : np.array
        array of all the classes found by the svc
    neighbours : list
        list of length 2 lists containing the eighbouring regions

    Returns
    -------
    list

    """    
    if len(neighbours) == 0 :
        for i in range(len(classes)) :
            for j in range(i+1, len(classes)) :
                neighbours += [[i, j]]
    
    # Order as a function of classes is details in https://scikit-learn.org/stable/modules/svm.html, Details on multi-class strategies
    distances = []
    existing_neighbours = []
    
    for n in neighbours :
        i0 = np.where(classes == n[0])[0]
        i1 = np.where(classes == n[1])[0]
        if (len(i0) > 0 and len(i1) > 0) :
            i0, i1 = min(i0[0], i1[0]), max(i0[0], i1[0])
        
            m = 0
            for j in range(i0) :
                m += len(classes)-j-1
                
            m += i1-i0-1
            
            distances.append(m)
            existing_neighbours.append([min(n), max(n)])
                            
    return(distances, existing_neighbours)

def parallel_random_(id, func, class_id) :
   
    x = func(id, class_id)
    return(x)

def parallel_random_eval_(id, func1, func2) :
    
    x = func1(id)
    y = func2(x)
    return([x, y])
    
def fit_(func, svc, X, y, a, b, neighbours) :
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
    clf = svm.SVC(**svc).fit((X-b)/a, y)
    clf._a, clf._b = a, b
    clf.__class__ = svcEDSD
    clf.trainingSet = np.array(X)
    clf.trainingSetValues = np.array(y)
    clf.func = func
    clf.svc = svc   
    clf.dimension_ = len(a)
    if (len(clf.classes_) > 2) :
        clf.decision_function_indices_, clf.neighbours_ = relevant_distances(clf.classes_, neighbours)
    
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
         verbose = True, svc={}, neighbours = []) :
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
    
    # Normalisation
    b = np.array(bounds[0])
    a = np.array(bounds[1])-np.array(bounds[0])
     
    bounds = [np.array(x) for x in bounds]    
   
    # Ajout de points aléatoires uniform dans bounds
    
    if verbose :
        print("Creation of first set")

    random.generate_pool(len(bounds[0]), N0)
    X += list(a*random.pool_+b)   
                
    # print(random.discrepency(X))
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
    if len(I) < len(X) and verbose :
        print("\nRemoving {} elements for class -1, stored in self.removed".format(len(X)-len(I)))
            
    removed = [X[i] for i in range(len(y)) if y[i] != -1]   

    X = np.array([X[i] for i in I])
    y = np.array([y[i] for i in I])
        
    # if classes > 2 :
    svc["decision_function_shape"]='ovo'
        
    clf = fit_(func, svc, X, y, a, b, neighbours)
    
    if verbose : 
        print("Classes found: ", clf.classes_)
        print("Number of points in each class:", [str(c) + ':' + str(len(np.where(y==c)[0])) for c in clf.classes_])
        
    
    clf = clf.expand(N1, processes = processes, verbose = verbose, neighbours = neighbours)

    clf.removed = removed

    # Get the bounds for all the classes in a dictionnary
    clf.classBounds = dict()
        
    for c in clf.classes_ :
        
        I = np.where(clf.trainingSetValues == c)[0]
                
        clf.classBounds[c] = [[min([clf.trainingSet[i][j] for i in I]) for j in range(len(clf.trainingSet[0]))], 
                                [max([clf.trainingSet[i][j] for i in I]) for j in range(len(clf.trainingSet[0]))]]  
                                      
    if verbose : 
        print("\nFinal set of classes : ", clf.classes_)
              
    return(clf)
    
    
