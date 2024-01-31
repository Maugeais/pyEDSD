import pyEDSD as edsd

clf = edsd.load("3d_multi.edsd")

edsd.set_backend("plotly")

# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# options = [{"name" : str(n)} for i, n in enumerate(clf._neighbours)]  

# fig = clf.draw(plot_method = "frontiers", scatter = True, options = options)

# fig.update_layout(
#     title="Example of EDSD in 3d",
#     scene = dict(xaxis_title='Example X',
#                     yaxis_title='Y Example',
#                     zaxis_title='Z test'),
#                     # width=700,
#                     # margin=dict(r=20, b=10, l=10, t=10),
#     legend_title="Legend Title",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="RebeccaPurple"
#     )
# )
    

# clf.show(fig)




options = {0 : {"name" : "Test 1", "showlegend" : True}, 
            1 : {"name" : "Toto 2", "showlegend" : True},
            2 : {"name" : "Titi 3", "showlegend" : True},
            3 : {"name" : "Titi 3", "showlegend" : True}}

fig = clf.draw(plot_method = "classes", scatter = True, classes = clf.classes_, options = options)

fig.update_layout(
    title="Example of EDSD in 3d",
    scene = dict(xaxis_title='Example X',
                    yaxis_title='Y Example',
                    zaxis_title='Z test'),
                    # width=700,
                    # margin=dict(r=20, b=10, l=10, t=10),
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
    
clf.show(fig)