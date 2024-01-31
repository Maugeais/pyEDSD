import sys
sys.path.append('../')

import pyEDSD as edsd

edsd.set_backend("plotly")
clf = edsd.load("3d.edsd")
fig = clf.draw(plot_method = "frontiers", contourf_options = [{"name" : "Test 1", "showlegend" : True, "color" : 'rgb(255,0,0)'}])
fig.update_layout(
    title="Example of 3d drawing",
    scene = dict(xaxis_title='Axis 1',
                    yaxis_title='Axis 2',
                    zaxis_title='Axis 3'),
                    # width=700,
                    # margin=dict(r=20, b=10, l=10, t=10),
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ),
    showlegend = True
)    
     
clf.show(fig)
