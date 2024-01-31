import pyEDSD as edsd

edsd.set_backend("plotly")
clf = edsd.load("3d.edsd")
fig = clf.draw(options = [{"name" : "Test 1", "showlegend" : True, "color" : 'rgb(255,0,0)'}])
# fig.update_layout(
#     title="Map of the trombone's regime",
#     scene = dict(xaxis_title='F',
#                     yaxis_title=r'$P_m$',
#                     zaxis_title='H'),
#                     # width=700,
#                     # margin=dict(r=20, b=10, l=10, t=10),
#     legend_title="Legend Title",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="RebeccaPurple"
#     ),
#     showlegend = True
# )    
     
clf.show(fig)
