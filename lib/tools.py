#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:45:18 2023

@author: maugeais
"""

import numpy as np
import warnings
import sys, os


def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

def volume(tri) :
    """
    

    Returns
    -------
    None.

    """
    
    vol = 0
    
    for simp in tri.simplices :

        M = []
        
        for p in simp[1:] :
            M.append(tri.points[p]-tri.points[simp[0]])

        # print(M)    
        vol += abs(np.linalg.det(M))
    
    if tri.points.shape[1] == 2 :
        return(vol/2)
    if tri.points.shape[1] == 3 :
        return(vol/6)
            

def _parallel(param, func1, func2=None, **kargs) :
    """
    Parallelize the function func1, and execute func2 on its result 
    if func2 is not None

    Parameters
    ----------
    param : int, optional
        Parameter to be given to the function func1
        The default is 0.
    func1 : function
        Function to be evaluated DESCRIPTION. The default is None.
    func2 : function, optional
        If not none, function to evaluate on the returning parameter of func1(param). 
        The default is None.

    Returns
    -------
    If func2 == None, returns func1(param)
    otherwize return [func1(param), func2(func1(param))]

    """
    print(param)
                
    x = func1(param, **kargs)
    if func2 != None :
        y = func2(x)
        return([x, y])
    
    else :
        return(x)

    
def _advBar(i) :
    """

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*(i//5), i))
    sys.stdout.flush()    
    
    
def convert2mp4(folder) :
    os.system("ffmpeg -r 10 -i "+folder+"img%05d.jpg -vcodec mpeg4 -y movie.mp4")       
    os.system("rm "+folder+"/tmp/img*.jpg")      
