import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

colors = list(mcolors.XKCD_COLORS.values())

try : 
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import plotly.express as px
    is_plotly = True
    
except ImportError:
    print("Plotly is not available on your system")
    is_plotly = False
    
    
_backend = "matplotlib"

# def _draw2d_(clf, grid_resolution = 100, scatter = True, fig = None, classes = [], ax = None, options ={}, levels = []) :
#     """
#     Draw the zones and their boundaries obtained for a 2d classifier
#     Parameters
#     ----------
#     grid_resolution : int, optional
#         DESCRIPTION. resolution of the grid used to draw the contour.
#         The default is 100.
#     scatter : bool, optional
#         DESCRIPTION. The default is False.
#     options : dict, optional
#         Options passed to DecisionBoundaryDisplay.from_estimator

#     Returns
#     -------
#     None.

#     """

#     X = clf.trainingSet
#     y = clf.trainingSetValues #predict((X-clf._b)/clf._a)
        
#     if len(clf._classes) == 2 :
#         levels = 0
#     else :
#         levels = len(clf._classes)-1
        
#     default = {'plot_method': "contourf",
#                 'levels' : levels}
        
#     disp = DecisionBoundaryDisplay.from_estimator(
#                 clf,
#                 (clf.trainingSet-clf._b)/clf._a,
#                 ax=ax,
#                 grid_resolution=grid_resolution,
#                 **{**default, **options},
#             )
#     # ax.set_aspect(1)
#     disp.ax_.set_xlim(0, 1)
#     disp.ax_.set_ylim(0, 1)
    
#     disp.ax_.xaxis.set_major_formatter(lambda x, pos : "{:.2f}".format(clf._a[0]*x+clf._b[0]))
#     disp.ax_.yaxis.set_major_formatter(lambda y, pos : "{:.2f}".format(clf._a[1]*y+clf._b[1]))

#     if scatter :
        
#         for i, c in enumerate(clf._classes) :
            
#             I = np.where(y == c)[0]
#             disp.ax_.scatter((X[I, 0]-clf._b[0])/clf._a[0], (X[I, 1]-clf._b[1])/clf._a[1], c=colors[i], marker = 'x', label="class "+str(c)) 
            
#         disp.ax_.legend()
        
#     return(disp.ax_)
    
def _contour3d(clf, grid_resolution = 50, scatter = True, classes = [], options ={}) :
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        
    X = clf.trainingSet
    y = clf.trainingSetValues
    
    
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
            
        return(clf.decision_function(np.c_[X, Y, Z]).reshape(ref))
    
    plot_implicit(ax, f, res=grid_resolution, bbox=[[0, 0, 0], [1, 1, 1]])
    
    if scatter : 
        ax.scatter((X[:, 0]-clf._b[0])/clf._a[0],(X[:, 1]-clf._b[1])/clf._a[1], (X[:, 2]-clf._b[2])/clf._a[2], c=y, cmap=plt.cm.coolwarm)
        
    plt.tight_layout()


def __prepare_grid(clf, grid_resolution) :

    if clf._dimension == 2 :
    
        # create a mesh to plot in
        x_min, x_max, y_min, y_max = 0, 1, 0, 1
        h = 1/grid_resolution
        xx, yy = np.mgrid[x_min:x_max:h, y_min:y_max:h]

        f = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

        xx, yy = clf._a[0]*xx+clf._b[0], clf._a[1]*yy+clf._b[1]

    else : 

        x_min, x_max, y_min, y_max, z_min, z_max = 0, 1, 0, 1, 0, 1
        h = 1/grid_resolution
        xx, yy, zz = np.mgrid[x_min:x_max:h, y_min:y_max:h, z_min:z_max:h]
    
        f = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

        xx, yy, zz = clf._a[0]*xx+clf._b[0], clf._a[1]*yy+clf._b[1], clf._a[2]*zz+clf._b[2]
                    
    if len(clf._classes) == 2 :
        
        F = [f.reshape(xx.shape)]        
        
    else :

        F = [f[:, c].reshape(xx.shape) for c in range(f.shape[1])]


    if clf._dimension == 2 :
        return(xx, yy, F)

    return(xx, yy, zz, F)


def _frontiers2d(clf, grid_resolution = 100, scatter = True, frontiers = [], 
                 ax = None, fig = None, options = [{}], label_options = [{}], scatter_options = [{}]) :
    """Draw the zones and their boundaries obtained for a 3d the classifier
    Parameters
    ----------
    grid_resolution : int, optional
        Resolution of the grid used to draw the contour.
        The default is 100.
    scatter : bool, optional
        The default is False.
    options : dictionnary, optional
        Options passed to the backend

    Returns
    -------
    None.

    """

    

    X = clf.trainingSet
    y = clf.trainingSetValues
            
    xx, yy, F = __prepare_grid(clf, grid_resolution)
                    
    if len(clf._classes)  > 2 :
        options += [options[-1]]*(len(clf._decision_function_indices)-len(options))

        label_options += [label_options[-1]]*(len(clf._decision_function_indices)-len(label_options))

        _neighbours = clf._neighbours
        
        if frontiers == [] :
                frontiers = clf._neighbours
    else :
        _neighbours = [0, 1]
        frontiers = [0]
            
    for i, f in enumerate(F) :
        
        if i in _neighbours and _neighbours[i] in frontiers and not (np.all(f > 0) if f[0, 0] > 0 else np.all(f < 0)) :
            
            if _backend == "plotly" :
                
                cmap = plt.get_cmap("tab10")
                
                colorscale = [[0, 'rgb' + str(cmap(1)[0:3])], 
                                [0.1, 'rgb' + str(cmap(2)[0:3])],
                                [0.2, 'rgb' + str(cmap(3)[0:3])],
                                [0.3, 'rgb' + str(cmap(4)[0:3])],
                                [1, 'rgb' + str(cmap(5)[0:3])]]
                
                
                default = {}
                            
                x, y, z = verts.T
                I, J, K = faces.T
                mesh = go.Mesh3d(x=x, y=y, z=z,
                        i=I,j=J,k=K, **{**default, **options[i]}, opacity=0.50)   
                
                if fig == None : 
                    fig = go.Figure()

                fig.add_trace(mesh)
       
            if _backend == "matplotlib" :
                
                if ax == None :
                    fig = plt.figure()
                    ax = plt.gca()

                default_options = {"levels" :  [0], "antialiased" : True}

                CS = ax.contour(xx, yy, f, **{**default_options, **options[clf._classes[i]]})

                default_options = {"inline" :  True, "fontsize" : 10}

                if "fmt" in label_options[i]:
                    ax.clabel(CS, CS.levels, **{**default_options, **label_options[i]})
            
                if scatter : 
                        ax.scatter(X[:, 0], X[:, 1], c=y)

                plt.tight_layout()
            
    if _backend == "matplotlib" :
        return(ax)

    if _backend == "plotly" :
        if scatter :
            fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers'))
            
        fig.update_traces()
        fig.update_layout(width=1200, height=1200, font_size=11)
        return(fig)


def _classes2d(clf, grid_resolution = 100, scatter = True, classes = [True], 
               ax = None, fig = None, options = [{}], label_options = [{}], scatter_options = [{}]) :
    """Draw the boundaries obtained for a 3d the classifier
    Parameters
    ----------
    grid_resolution : int, optional
        Resolution of the grid used to draw the contour.
        The default is 100.
    scatter : bool, optional
        The default is False.
    options : dictionnary, optional
        Options passed to the backend

    Returns
    -------
    None.

    """
    X = clf.trainingSet
    y = clf.trainingSetValues
            
    xx, yy, F = __prepare_grid(clf, grid_resolution)

    if len(clf._classes) > 2 :
    
        neighbourhood = set.union(*(set(x) for x in clf._neighbours))

    else :
        neighbourhood = {True, False}
        clf._neighbours = [[True, False]]

    
    for c in classes :

        if c in neighbourhood  :           
            
            f = np.max([(-float(c==n[0])+float(c==n[1]))*F[clf._index_neighbourhood(n)] for n in clf._neighbours if c in n], axis = 0)

            if _backend == "matplotlib" :
                
                if ax == None :
                    fig = plt.figure()
                    ax = plt.gca()

                default_options = {"levels" :  [-1000, 0], "antialiased" : True}
                CS = ax.contourf(xx, yy, f, **{**default_options, **options[c]})
                pc =  CS.collections[0]

                default_options = {"fc" : pc.get_facecolor()[0]}
                rect = patches.Rectangle((0,0),0,0, **{**default_options, **label_options[c]})

                ax.add_patch(rect)


    plt.legend()
           
    if _backend == "matplotlib" :
        
        if scatter and ax != None : 
            
            for c in classes : 
                I = np.where(y == c)[0]
                ax.scatter(X[I, 0], X[I, 1], **scatter_options[c])
            
        return(ax)

    # if _backend == "plotly" :
    #     if scatter :
    #         fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers'))
            
    #     fig.update_traces()
    #     fig.update_layout(width=1200, height=1200, font_size=11)
    #     return(fig)
  
def _frontiers3d(clf, grid_resolution = 100, scatter = True, frontiers = [], 
                 ax = None, fig = None, options = [], label_options = [{}], scatter_options = [{}]) :
    """Draw the zones and their boundaries obtained for a 3d the classifier
    Parameters
    ----------
    grid_resolution : int, optional
        Resolution of the grid used to draw the contour.
        The default is 100.
    scatter : bool, optional
        The default is False.
    options : dictionnary, optional
        Options passed to the backend

    Returns
    -------
    None.

    """

    X = clf.trainingSet
    y = clf.trainingSetValues
            
    xx, yy, zz, F = __prepare_grid(clf, grid_resolution)

    h = 1/grid_resolution

    if len(clf._classes)  > 2 :
        options += [options[-1]]*(len(clf._decision_function_indices)-len(options))

        label_options += [label_options[-1]]*(len(clf._decision_function_indices)-len(label_options))

        _neighbours = clf._neighbours
        if frontiers == [] :
                frontiers = clf._neighbours
    else :
        _neighbours = [0]
        frontiers = [0]
        
    for i, f in enumerate(F) :

        if _neighbours[i] in frontiers and not (np.all(f > 0) if f[0, 0, 0] > 0 else np.all(f < 0)) :
            
            verts, faces, normals, values = measure.marching_cubes(f, 0, spacing=[h, h, h])
            verts = clf._a*verts+clf._b
            
            if _backend == "plotly" :
                
                cmap = plt.get_cmap("tab10")
                
                colorscale = [[0, 'rgb' + str(cmap(1)[0:3])], 
                                [0.1, 'rgb' + str(cmap(2)[0:3])],
                                [0.2, 'rgb' + str(cmap(3)[0:3])],
                                [0.3, 'rgb' + str(cmap(4)[0:3])],
                                [1, 'rgb' + str(cmap(5)[0:3])]]
                
                
                default = {}
                            
                x, y, z = verts.T
                I, J, K = faces.T
                mesh = go.Mesh3d(x=x, y=y, z=z,
                        i=I,j=J,k=K, **{**default, **options[i]}, opacity=0.50)   
                
                if fig == None : 
                    fig = go.Figure()

                fig.add_trace(mesh)
    
                
                
            if _backend == "matplotlib" :
                
                if ax == None :
                    fig = plt.figure()
                
                    ax = fig.add_subplot(111, projection='3d')

                default_options = {"alpha" : 0.5, "lw" : 0, "antialiased" : True}
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], **{**default_options, **options[i]})    
                
    if _backend == "matplotlib" :
        if scatter : 
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
    
        return(ax)

    if _backend == "plotly" :
        if scatter :
            fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers'))
            
        fig.update_traces()
        fig.update_layout(width=1200, height=1200, font_size=11)
        return(fig)

def _classes3d(clf, grid_resolution = 100, scatter = True, classes = [], 
               ax = None, fig = None, options = [], levels = [], 
               label_options = [{}], scatter_options = [{}]) :
    """Draw the zones and their boundaries obtained for a 3d the classifier
    Parameters
    ----------
    grid_resolution : int, optional
        Resolution of the grid used to draw the contour.
        The default is 100.
    scatter : bool, optional
        The default is False.
    options : dictionnary, optional
        Options passed to the backend

    Returns
    -------
    None.

    """

    X = clf.trainingSet
    Y = [list(clf._classes).index(i) for i in clf.trainingSetValues]

    if (len(clf._classes)) == 0 :
        classes = clf._classes
            
    xx, yy, zz, F = __prepare_grid(clf, grid_resolution)

    h = 1/grid_resolution

    neighbourhood = set.union(*(set(x) for x in clf._neighbours))

    for c in classes :
        
        if c in neighbourhood  :
                    
            f = np.max([(-float(c==n[0])+float(c==n[1]))*F[clf._index_neighbourhood(n)] for n in clf._neighbours if c in n], axis = 0)
            
            if not (np.all(f > 0) if f[0, 0, 0] > 0 else np.all(f < 0)) :
             
                verts, faces, normals, values = measure.marching_cubes(f, 0, spacing=[h, h, h])
                verts = clf._a*verts+clf._b
                
                if _backend == "plotly" :
                    
                    cmap = plt.get_cmap("tab10")
                    
                    colorscale = [[0, 'rgb' + str(cmap(1)[0:3])], 
                                    [0.1, 'rgb' + str(cmap(2)[0:3])],
                                    [0.2, 'rgb' + str(cmap(3)[0:3])],
                                    [0.3, 'rgb' + str(cmap(4)[0:3])],
                                    [1, 'rgb' + str(cmap(5)[0:3])]]
                    
                    
                    default = {}
                                
                    x, y, z = verts.T
                    I, J, K = faces.T
                    mesh = go.Mesh3d(x=x, y=y, z=z,
                            i=I,j=J,k=K, **{**default, **options[c]}, opacity=0.50)   
                    
                    if fig == None : 
                        fig = go.Figure()
    
                    fig.add_trace(mesh)
    
                
                
                if _backend == "matplotlib" :
                    
                    if ax == None :
                        fig = plt.figure()
                    
                        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
                    default_options = {"alpha" : 0.5, "lw" : 0.0, "antialiased" : False}
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], **{**default_options, **options[c]})
                
    if _backend == "matplotlib" :
        if scatter : 
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.coolwarm)
        return(ax)

    if _backend == "plotly" :
        if scatter :
            fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers'))
            
        fig.update_traces()
        fig.update_layout(width=1200, height=1200, font_size=11)
        return(fig)
 

def show(fig = None) :
    if _backend == "plotly" :
        fig.show()
    
    if _backend == "matplotlib" :
        plt.show()


def plot_implicit(ax, fn, res = 15, bbox=[[0, 0, 0], [1, 1, 1]]):
    """
    Create a plot of an implicit function using the contour

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    fn : function return a float
        implicit function (plot where fn==0)
    bbox : TYPE, optional
        the x,y,and z limits of plotted interval

    Returns
    -------
    None.

    """
    xmin, ymin, zmin = bbox[0]
    xmax, ymax, zmax = bbox[1]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No contour levels were found within the data range.") 
    
        Ax = np.linspace(xmin, xmax, res) # resolution of the contour
        Ay = np.linspace(ymin, ymax, res) # resolution of the contour
        Az = np.linspace(zmin, zmax, res) # resolution of the contour

        X,Y = np.meshgrid(Ax,Ay) # grid on which the contour is plotted
        for z in Az: # plot contours in the XY plane
            Z = fn(X,Y,z)
            cset = ax.contour(X, Y, Z+z, [z], zdir='z')
            # [z] defines the only level to plot for this contour for this value of z
    
        X, Z = np.meshgrid(Ax,Az) # grid on which the contour is plotted
        for y in Ay: # plot contours in the XZ plane
            Y = fn(X,y,Z)
            cset = ax.contour(X, Y+y, Z, [y], zdir='y')
    
        Y, Z = np.meshgrid(Ay,Az) # grid on which the contour is plotted
        for x in Ax: # plot contours in the YZ plane
            X = fn(x,Y,Z)
            cset = ax.contour(X+x, Y, Z, [x], zdir='x')
    

    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)
    
def triplot(x, y, s, *args) :
    plt.triplot(x, y, s, *args)
