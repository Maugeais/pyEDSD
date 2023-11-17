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
import scipy.spatial as spatial


from sklearn.inspection import DecisionBoundaryDisplay
import multiprocessing
import time
import sys, pickle


from skimage import measure

colors = ['r', 'g', 'b', 'c', 'm', 'k']
import warnings

warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

def plot_implicit(ax, fn, bbox=(-2.5,2.5)):
    """
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    fn : TYPE
        DESCRIPTION.
    bbox : TYPE, optional
        DESCRIPTION. The default is (-2.5,2.5).

    Returns
    -------
    None.

    """
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3

    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted
    
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No contour levels were found within the data range.") 
   

        for z in B: # plot contours in the XY plane
            X,Y = A1,A2
            Z = fn(X,Y,z)
            cset = ax.contour(X, Y, Z+z, [z], zdir='z')
            # [z] defines the only level to plot for this contour for this value of z
    
        for y in B: # plot contours in the XZ plane
            X,Z = A1,A2
            Y = fn(X,y,Z)
            cset = ax.contour(X, Y+y, Z, [y], zdir='y')
    
        for x in B: # plot contours in the YZ plane
            Y,Z = A1,A2
            X = fn(x,Y,Z)
            cset = ax.contour(X+x, Y, Z, [x], zdir='x')


    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)
    
    
class svcEDSD(svm.SVC):
    
    def draw(self, scatter=True, grid_resolution = 100):
        
        if (len(self.bounds[0]) == 2) : 
            self.draw2d(scatter=scatter, grid_resolution = grid_resolution)
            
        elif (len(self.bounds[0]) == 3) : 
            self.draw3d(scatter=scatter, grid_resolution = grid_resolution)
            # self.contour3d(scatter=scatter, grid_resolution = grid_resolution)


    def draw2d(self, grid_resolution = 100, scatter = True) :
        """
        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
            
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
        # ax.set_aspect(1)
        plt.xlim(self.bounds[0][0], self.bounds[1][0])
        plt.ylim(self.bounds[0][1], self.bounds[1][1])
        if scatter :
            
            for i, c in enumerate(self.classes_) :
                
                I = np.where(y == c)[0]
                plt.scatter(X[I, 0], X[I, 1], c=colors[i], cmap=plt.cm.coolwarm, marker = 'x', alpha =0.5, label="class "+str(c)) 
                
        plt.legend()
        
    def contour3d(self, grid_resolution = 10, scatter = True) :
        """
        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
            
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
        
        # Il faut aussi renormaliser !!!!

        def f(x,y,z):
            
            if type(x) != np.ndarray :
                Z = z.ravel()
                Y = y.ravel()
                X = x+0*Z
                ref = z.shape
                
            if type(y) != np.ndarray :
                X = x.ravel()
                Z = z.ravel()
                Y = y+0*X
                ref=  x.shape
                
            if type(z) != np.ndarray :
                X = x.ravel()
                Y = y.ravel()
                Z = z+0*X
                ref=  x.shape
                
            W = self.decision_function(np.c_[X, Y, Z]).reshape(ref)
            return W
        
        ax = plt.gca()
        plot_implicit(ax, f)
                
        if scatter : 
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
        plt.tight_layout()
        
    def draw3d(self, grid_resolution = 10, scatter = True) :
        """
        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
            
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
        xx, yy, zz = np.mgrid[bounds[0][0]:bounds[1][0]:hx, bounds[0][1]:bounds[1][1]:hy, bounds[0][2]:bounds[1][2]:hz]
        
        
        
        F = self.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()]).reshape(xx.shape)

        verts, faces, normals, values = measure.marching_cubes(F, 0, spacing=[hx, hy, hz])
        verts -= bounds[1]
        

        
        ax = plt.gca()
        
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, lw=0, antialiased=True)
        
        if scatter : 
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
        plt.tight_layout()
        
        
    def chgRandomBounds(self, X, error = 0.05) :
        """ Change the bounds for the random operations
        
            if X is a set of coordinates then
            if X is a class then
            
            the error term is used .... """
        """
        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

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
        
        """ Main random function, it should not be called directly"""
        """
        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

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
                print("Error in the generation of the random points, try ...")
                raise Exception('random')

    def resetRandomPool(self) :
        
        self._randomPool = []
        
    def random(self, size=1, processes = 1, verbose = False) :
        """ PArallelisation of the random function, and fills the random pool"""
        """
        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if size == 1 : 
            
            return(self._random())
      
        if not hasattr(self, '_randomPool') :
            
            self._randomPool = []
            
        newSize = max(size-len(self._randomPool), 0)
        
        n = 0
         
        with multiprocessing.Pool(processes=processes) as pool:
                    
            for r in pool.map(partial(_parallel_, rand=self._random), range(newSize)):
                    
                self._randomPool.append(r)

                if verbose : 
                    n += 1
                    advBar(int(100*n/newSize))  
                    
        if verbose :
            print("")
                
        I = np.random.randint(0, len(self._randomPool), size=size)
                
        return([self._randomPool[i] for i in I])
    
    def diameter_estimate(self, size_random = None) :
        """
        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
        
        X = self.random(size=size_random)
        
        return(max(spatial.distance.pdist(X)))
    
    def boundingbox_estimate(self, size_random = None) :
        """

        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
    
        X = np.array(self.random(size=size_random))
        
        return([np.array([min(X[:, 0]), min(X[:, 1])]), np.array([max(X[:, 0]), max(X[:, 1])])])
    
    def distFrom(self, P=[], size_random=None) :
        """

        Parameters
        ----------
        P : TYPE, optional
            DESCRIPTION. The default is [].
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
    
        X = np.array(self.random(size=size_random))
        
        plt.scatter(*P)
        return(min(np.linalg.norm(X-np.array(P), axis=1)))
    
    
    def volumeMonteCarlo(self, size_random=None) :
        """ Très mauvais, car la convergence est très lente !!!

        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
            
        bounds = self.bounds #self.boundingbox_estimate()
    
        X = bounds[0]+(bounds[1]-bounds[0])*np.random.rand(size_random, len(self.bounds[0]))
        
        y = self.predict(X)
        
        V = np.prod([bounds[1][i]-bounds[0][i] for i in range(len(bounds[0]))])
        
        return (V*(1-y.sum()/len(y)))
    
    def delaunay(self, size_random = None) :
        """

        Parameters
        ----------
        size_random : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if size_random == None :
            size_random = 10**len(self.bounds[0])
            
        
        points = self.random(size_random)
        
        points = np.vstack(points)
        
    
        self.triangulation = spatial.Delaunay(points)
        
        
        # Remove simplicies whose centre is outside 
        
        I = [simp for simp in list(self.triangulation.simplices) if self.decision_function([np.mean(self.triangulation.points[simp], axis = 0)]) < 0]
    
        self.triangulation.simplices = I    


from functools import partial 
def _parallel_(id = 0, rand=None, func=None) :
                
    x = rand(id = id)
    if func != None :
        y = func(x)
        return([x, y])
    
    else :
        return(x)

def advBar(i) :
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*(i//5), i))
    sys.stdout.flush()  
    
    
def save(clf, filename) :
    with open(filename, 'wb') as file:
        pickle.dump(clf, file) 

def load(filename) :
    with open(filename, 'rb') as file:
        res = pickle.load(file)     
    return(res)
    
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
                advBar(int(100*n/len(X)))
        
    if classes > 2 :
        svc["decision_function_shape"]='ovo'
        
        
    clf = svm.SVC(**svc).fit(X, y)
    clf.bounds = bounds
    clf.__class__ = svcEDSD
    clf.trainingSet = np.array(X)
    
    if verbose : 
        print("\nClasses found : ", clf.classes_)
        print("Number of points in each class :", [str(c) + ':' + str(len(np.where(y==c)[0])) for c in clf.classes_])
            

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
            
            try :
            
                with multiprocessing.Pool(processes=processes) as pool:
                    
                    for r in pool.map(partial(_parallel_, rand=clf._random, func=func), range(processes)):
                        
                        X.append(r[0])
                        y.append(r[1])
#             except KeyboardInterrupt:
#                 
#                 print('houla', "fin")
            # except Exception as e:
            #     print('toto', e)
            except BaseException as error:
                
                print(error)
                clf = svm.SVC(**svc).fit(X, y)
                clf.bounds = bounds
                clf.__class__ = svcEDSD
                clf.trainingSet = np.array(X)
                
                break
                
        clf = svm.SVC(**svc).fit(X, y)
        clf.bounds = bounds
        clf.__class__ = svcEDSD
        clf.trainingSet = np.array(X)
        clf.trainingSetValue = np.array(y)
        
        
        
        if animate :
            
            clf.draw()

            plt.savefig('/tmp/img'+format(n, '05d')+'.jpg', dpi=200)
            ax.clear()
            
        if verbose : 
            advBar(int(100*n*processes/N1))
        
        n += 1
        
        
    # Get the bounds for all the classes in a dictionnary
    clf.classBounds = dict()
        
    for c in clf.classes_ :
        
        I = np.where(y == c)[0]
                
        # clf.classBounds[c] = [np.array([min([X[i][j] for i in I]), max([X[i][0] for i in I])]) for j in range(len(X[0]))]
        # a = np.array([min([X[i][j] for j in range(len(X[0]))]) for i in I])
        # print(a)
        clf.classBounds[c] = [np.array([min([X[i][j] for i in I]) for j in range(len(X[0]))]), 
                                np.array([max([X[i][j] for i in I]) for j in range(len(X[0]))])]  
                                      
    if verbose : 
        print("\nFinal set of classes : ", clf.classes_)
    
            
    if animate :
        import os 
        os.system("ffmpeg -r 10 -i /tmp/img%05d.jpg -vcodec mpeg4 -y movie.mp4")       
        os.system("rm /tmp/img*.jpg")       
              
    return(clf)
    
    
