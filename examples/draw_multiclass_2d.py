#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import pyEDSD as edsd
import matplotlib.pyplot as plt
import numpy as np
clf = edsd.load("multi.edsd")

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

options = {0 : {"colors" : 'w'},
                    1 : {"colors" : 'c'},
                    2 : {"colors" : 'm'},
                    3 : {"colors" : 'b'},
                    4 : {"colors" : 'g'},
                    }

label_options =  {0 : {"label" : "Regime 0"},
                    1 : {"label" : "Regime 1"},
                    2 : {"label" : "Regime 2"},
                    3 : {"label" : "Regime 3"},
                    4 : {"label" : "Regime 4"},
                    }
    
ax = clf.draw(plot_method = "classes", scatter = False, classes = [0, 1, 2, 3, 4, 5], options = options, label_options = label_options)    

X = clf.random(size = 100, processes = 5, class_id=2)
X = np.array(X)

plt.scatter(X[:, 0], X[:, 1])
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# options = [{"colors" : colors[i % len(colors)]} for i, n in enumerate(clf.neighbours_)]  

# label_options = [{"fmt" : str(n)} for i, n in enumerate(clf.neighbours_)]  

# clf.draw(plot_method = "frontiers", scatter = False, options = options, label_options = label_options, ax = ax)
clf.show()
    