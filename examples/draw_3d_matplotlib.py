import sys
sys.path.append('../')

import pyEDSD as edsd
import matplotlib.pyplot as plt
import numpy as np

edsd.set_backend("matplotlib")
clf = edsd.load("3d.edsd")
ax = clf.draw(plot_method = "frontiers", options = [{"color":[0.1, 0.1, 0.1, 0.5]}])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
clf.show()
