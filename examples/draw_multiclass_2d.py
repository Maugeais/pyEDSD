#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:56:45 2023

@author: maugeais
"""

import pyEDSD as edsd

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
    
clf.draw(plot_method = "classes", scatter = False, classes = [0, 1, 2, 3, 4, 5], options = options, label_options = label_options)    


colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

options = [{"colors" : colors[i % len(colors)]} for i, n in enumerate(clf._neighbours)]  

label_options = [{"fmt" : str(n)} for i, n in enumerate(clf._neighbours)]  

clf.draw(plot_method = "frontiers", scatter = False, options = options, label_options = label_options)
clf.show()
    