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

import multiprocess
import time
import pickle

from lib import plot, tools

__version__ = "0.5.3.1"

from functools import partial 
import scipy.spatial as spatial


    
plot._backend = "matplotlib"

def set_backend(backend = "matplotlib") :
    """
    Changes the backend used for plotting 

    Parameters
    ----------
    backend : string, optional
        Name of the backend to use, "matplotlib" or "plotly". 
        The default is "matplotlib".

    Returns
    -------
    None.

    """    
    if backend in ["matplotlib", "plotly"] :
        plot._backend = backend
    else :
        print(f"The backend '{backend}' for is not supported")
        
def identity(x) :
    return(x)

max_random_gradient = 100 # Maximal number of unconclusive gradient iteration

class svcEDSD(svm.SVC):
    
    def draw(self, plot_method = "frontiers", grid_resolution = 100, scatter = False, classes = [], frontiers = [], ax = None, fig = None, options = [{}], label_options = [{}], scatter_options = [{}]):
        """
        Draw the zones and their boundaries obtained by the classifier using 
        either matplolib or plotly depending on the backend fixed by set_backend.
        
        
        Parameters
        ----------
        plot_method : string, optional
            Set the type of plot that has to be drawn, depending on the backend
        
            If backend == "matplotlib" (default)
            
                * If the dimension is 2
            
                    if plot_method == 'frontiers' 
                        - option is passed to "contour" (cf. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)
                        - label_options is passed to  "clabel" (cf. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.clabel.html)
                        - scatter_options is not yet implemented
                        
                    if plot_method == 'classes' 
                        - option is passed to "contourf" (cf. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)
                        - label_options is passed to  "Rectangle" (cf. https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html)
                        - scatter_options is passed to "scatter" (cf. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter)
                    
                * If the dimension is 3
                    
                    - option is passed to plot_trisurf (cf. https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf)
                    - label_options is not yet implemented
                    - scatter_options is not yet implemented
                        
            If backend == "plotly" (default)
            
                * If the dimension is 2
                    
                    Not yet implemented
            
                * If the dimension is 3
            
                    - option is passed to Mesh3d (cf. https://plotly.com/python/reference/mesh3d/)
                    - scatter_options is not yet implemented 
                                    
        grid_resolution : int, optional
            resolution of the grid used to draw the contour.
            The default is 100.
        scatter : bool, optional
            The default is False.
        classes  : list, optional, 
            Set of classes to be drawn, only relevvant if plot_method = "classes"
        frontiers : list, optional, 
            Set of frontiers to be drawn, only relevvant if plot_method = "frontiers"
        ax : matplotlib.axes or None, default is None
            Axis to be used to draw when backend is "matplotlib". If None, a new axis is created
        fig : Figure object or None, default is None
            Figure to be used to draw when backend is "plotly". If None, a figure is created

        Returns
        -------
        In case backend == "matplotlib", returns the axis used to draw, and in case "plotly" returns the figure.

        """        

        if (self._dimension == 2) : 
            if plot_method == 'frontiers' : 
                return(plot._frontiers2d(self, grid_resolution = grid_resolution, scatter = scatter, frontiers = frontiers, ax = ax, 
                                       options = options, label_options = label_options, scatter_options = scatter_options))
            
            elif plot_method == "classes" :
                return(plot._classes2d(self, grid_resolution = grid_resolution, scatter = scatter, classes = classes, ax = ax, 
                            options = options, label_options = label_options, scatter_options = scatter_options))
            
            
            else :
                raise ValueError(f"the value {plot_method} is not possible for the vairable plot_method")
                
            
        elif (self._dimension == 3) : 

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
        """
        

        Parameters
        ----------
        fig : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        plot.show(fig)
        
    def _index_neighbourhood(self, n) :
        
        i0 = np.where(self._classes == n[0])[0][0]
        i1 = np.where(self._classes == n[1])[0][0]
        i0, i1 = min(i0, i1), max(i0, i1)
                     
        m = 0
        for j in range(i0) :
            m += len(self._classes)-j-1
            
        m += i1-i0-1              
        return(m)
    
    def restriction(self, bounds = None, restriction_function = identity) :
        """
        

        Parameters
        ----------
        bounds : list of 2 n-dimensionnal vector, or None
            Indicating the lower and lower bounds for the restricted clf
            If None, the bounds are kept as the parent's bounds
        restriction_function : function
            Mapping from the restricted set to the parent definition set.
        
        Returns
        -------
        svcEDSD

        """
        
        clf = svcEDSD()

        if bounds == None :
            clf._a, clf._b = self._a, self._b
        else :
            clf._a, clf._b = np.array(bounds[1])-np.array(bounds[0]), np.array(bounds[0])

        clf.trainingSet = []
        clf.trainingSetValues = []
        clf._classes = self._classes
        if (len(clf._classes) > 2) :
            clf._decision_function_indices = self._decision_function_indices
            clf._neighbours = self._neighbours
        clf._dimension = len(clf._a) 
        clf.restriction_function = restriction_function
        clf.decision_function = clf.restricted_decision_function
        clf.parent = self        
        return(clf)
        
    def restricted_decision_function(self, x) :
        """
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """        
        y = self.parent.decision_function((np.array([self.restriction_function(a*self._a+self._b) for a in x])-self.parent._b)/self.parent._a)
        return(y)


    def _choose_multi_decision_function(self, class_id = -1) :
        
        if type(class_id) == int and class_id < 0 :
            classNumber = np.random.choice(self._decision_function_indices)
            decision = lambda x : self.decision_function([x])[0][classNumber]**2 #+1/(1+min(np.linalg.norm(self.trainingSet-x, axis=1)**2))
        
        else :
                                
            decision = lambda x : self.class_decision_function([x], class_id=class_id)**2
            
        return(decision)
        
    def _random(self, id=0, class_id = -1) :
        """
        Main random function, it should not be called directly

        Parameters
        ----------
        id : int, optional
            parameter used to initialise the random seed, together with the clock.
            This is present to prevent many calls to _random sent at the same time 
            that would  give the same result, which can happen if the clock resolution 
            used for the initialization of the seed is too coarse.
            The default is 0.

        Raises
        Exception 'random'
            If a new point has not been found within the boundaries after a 100 trials

        Returns
        -------
        A point close to the one of the boundaries of the classifier

        """
        tol = 1e-3
        
        np.random.seed((2**3*id+time.time_ns())%(2**32))
            
        if len(self._classes) > 2 :
            
            decision = self._choose_multi_decision_function(class_id = class_id) 
                
        else :
            
            decision = lambda x : self.decision_function([x])**2 #+1/(1+dist(x, X)**2)
            
        n = m = 0

        x0 = np.random.random(self._dimension)
        
        while True : 
                 
            res = minimize(decision, x0, method='CG', options = {})#, bounds=bounds)
            
            # If the result is within the bounds
            if  res.fun < tol and all([(res.x[i] > 0) and (res.x[i] < 1) for i in range(self._dimension)]) :
                return(self._a*res.x+self._b)         
            
            if n > max_random_gradient :

                m += 1
                                               
                if len(self._classes) == 2 or (type(class_id) != int) or m > len(self._classes) :
                    
                    raise ValueError('Random')              
                                
                decision = self._choose_multi_decision_function(class_id = class_id) 

                n = 0

            # if it did not work with original pool, try another random element
            n += 1

            x0 = np.random.random(self._dimension)

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
            
        #     self._randompool = []
        
        self._random_pool = []
        
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
               
        if not hasattr(self, '_random_pool') :
            
            self._random_pool = []
               
        new_size = max(size-len(self._random_pool), 0)
        
        if size == 1 :     
            return(self._random())
        
        n = 0
         
        try :
    
            with multiprocess.Pool(processes=processes) as pool:
                        
                for r in pool.map(partial(_parallel_random, func = self._random, class_id = class_id), range(new_size)) :
                        
                    self._random_pool.append(r)

                    if verbose : 
                        n += 1
                        tools._advBar(int(100*n/new_size))  

        except Exception as e :

            if repr(e) == "ValueError('Random')" :
                pass
        
            else :
                raise


        if verbose :
            print("")
                
        if len(self._random_pool) > 0 :
            I = np.random.randint(0, len(self._random_pool), size=size)
                    
            return(np.array([self._random_pool[i] for i in I]))
        else :
            return([])

    def set_random_box(self, bounds) :
        _a, _b = np.array(bounds[1])-np.array(bounds[0]), np.array(bounds[0])
        clf = _fit(self.func, self.svc, self.trainingSet, self.trainingSetValues, _a, _b, None)
        return(clf)




    
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
            size_random = 10**self._dimension
        
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
            size_random = 10**self._dimension
                
        X = np.array(self.random(size=size_random, class_id = class_id, processes = processes))    

        if len(X) > 0 :
            return([np.array([min(X[:, i]) for i in range(self._dimension)]), np.array([max(X[:, i]) for i in range(self._dimension)])])
        else :
            return([[0 for i in range(self._dimension)], [0 for i in range(self._dimension)]])
          
    
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
            size_random = 10**self._dimension
    
        X = np.array(self.random(size=size_random, class_id = class_id, processes = processes))
        
        plot.scatter(*P)
        return(min(np.linalg.norm(X-np.array(P), axis=1)))
    
    def delaunay(self, class_id = -1, frontier = [0, 1], n_boundary = 0, n_interior = 10, draw = False, ax = None, processes = 1) :
        """
        Compute a delaunay triangulation one region given by the classifier .
        For this, a delaunay triangulation is computing using :
        - the points of the training set
        - points on the boundary, using the random function
        - points in the interior, obtained using n_interior**dim points chosen randomly within the boundaries, and sorted using the classifier
                
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
    
        # Creates the points on the boundary
        points = self.random(n_boundary, class_id = class_id, processes = processes)

        m, M = np.min(points, axis = 0), np.max(points, axis = 0)
        
        # Then the points in the interior
        n_interior = max(n_interior, 2**self._dimension)

        rand = (M-m)*np.random.random((n_interior, self._dimension))+m
 
        if len(self._classes) == 2 :
                        
            decision = self.homogenous_decision_function
            
        else :
        
            decision = lambda x : self.homogenous_class_decision_function(x, class_id=class_id)
            
        I = decision(rand) < 0           
        
        points = np.concatenate((points, rand[I, :]))

        tri = spatial.Delaunay((points-self._b)/self._a)
        tri.points[:] = self._a*tri.points[:] + self._b
        
                
        tri.simplices = [simp for simp in list(tri.simplices) if decision([np.mean(tri.points[simp], axis = 0)]) < 0]
        
        if draw :
            
            if self._dimension == 2 :
                        
                ax.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices)
     
        return(tri)
    
        
    
    def volume(self, class_id, n_boundary = 100, n_interior = 500, processes = 1) :
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
        res = np.max([(-float(class_id==n[0])+float(class_id==n[1]))*self.decision_function(x)[:, self._index_neighbourhood(n)] for n in self._neighbours if class_id in n], axis = 0)
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
                
                x0 = clf._random()
                X.append(x0)
                y.append(self.func(x0))
            
            else :         
 
                try :
                
                    with multiprocess.Pool(processes=processes) as pool:
                        
                        for r in pool.map(partial(_parallel_randomeval, func1=clf._random, func2=self.func), range(processes)):

                            if r[1] != -1 :
                                X.append(r[0])
                                y.append(r[1])
                                
                except KeyboardInterrupt:
                    print('\nStopping at {}%'.format(100*n*processes/N1))
                    clf = _fit(self.func, self.svc, X, y, self._a, self._b, neighbours)
                    break
                
                except Exception as e :

                    if repr(e) == "ValueError('Random')" :
                    
                        clf = _fit(self.func, self.svc, X, y, self._a, self._b, neighbours)
                
                    else :
                        raise

            clf = _fit(self.func, self.svc, X, y, self._a, self._b, neighbours)    
                
            if verbose : 
                tools._advBar(int(100*(n+1)*processes/N1))
            
            n += 1
                                        
        return(clf)

def _relevant_distances(classes, neighbours) :
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
                neighbours += [[classes[i], classes[j]]]
    
    # Order as a function of classes is details in https://scikit-learn.org/stable/modules/svm.html, Details on multi-class strategies
    distances = []
    existing_neighbours = []
    
    for n in neighbours :

        if (n[0] in classes) and (n[1] in classes) :
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

def _parallel_random(id, func, class_id) :
   
    x = func(id, class_id)
    return(x)

def _parallel_randomeval(id, func1, func2) :
    
    x = func1(id)
    y = func2(x)
    return([x, y])
    
def _fit(func, svc, X, y, a, b, neighbours) :
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
    clf._classes = clf.classes_
    clf._a, clf._b = a, b
    clf.__class__ = svcEDSD
    clf.trainingSet = np.array(X)
    clf.trainingSetValues = np.array(y)
    clf.func = func
    clf.svc = svc   
    clf._dimension = len(a)
    if (len(clf._classes) > 2) :
        clf._decision_function_indices, clf._neighbours = _relevant_distances(clf._classes, neighbours)
    
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
        
    processes = min(multiprocess.cpu_count(), processes)
    
    # Normalisation
    b = np.array(bounds[0])
    a = np.array(bounds[1])-np.array(bounds[0])
     
    bounds = [np.array(x) for x in bounds]    
   
    
    if verbose :
        print("Creation of first set")
 
    X += list(a*np.random.rand(N0, len(bounds[0]))+b)
                
    # Calcul des valeurs des fonctions
    n = 0        
    with multiprocess.Pool(processes=processes) as pool:
                
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
        
    clf = _fit(func, svc, X, y, a, b, neighbours)
    
    if verbose : 
        print("\n Classes found: ", clf._classes)
        print("Number of points in each class:", [str(c) + ':' + str(len(np.where(y==c)[0])) for c in clf._classes])
        
    clf = clf.expand(N1, processes = processes, verbose = verbose, neighbours = neighbours)

    clf.removed = removed

    # Get the bounds for all the classes in a dictionnary
    clf.classBounds = dict()
        
    for c in clf._classes :
        
        I = np.where(clf.trainingSetValues == c)[0]
                
        clf.classBounds[c] = [[min([clf.trainingSet[i][j] for i in I]) for j in range(len(clf.trainingSet[0]))], 
                                [max([clf.trainingSet[i][j] for i in I]) for j in range(len(clf.trainingSet[0]))]]  
                                      
    if verbose : 
        print("\nFinal set of classes : ", clf._classes)
              
    return(clf)
    
    
