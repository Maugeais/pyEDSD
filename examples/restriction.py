import pyEDSD as edsd

clf = edsd.load("multi_3d.edsd")

""" The restriction function can be called with a function lambda"""

lambda_clf = clf.restriction([[-4, -4], [4, 4]], lambda x:[x[0], 0.5, x[1]])


options = {"0" : {"colors" : "r"}, 
            "1" : {"colors" : "g"},
            "2" : {"colors" : "b"},
            "3" : {"colors" : "c"}}

label_options = {"0" : {"label" : "Test 1"}, 
            "1" : {"label" : "Toto 2"},
            "2" : {"label" : "Titi 3"},
            "3" : {"label" : "Titi 3"}}

lambda_clf.draw(plot_method = "classes", scatter = False, classes = ["0", "1", "2"], options = options, label_options = label_options)

clf.show()


""" But as lambdas are not pickable, random points cannot be generated with it. 
    Instead, replace the lambda by a real function """

def fun(x) :
    return([x[0], 0.5, x[1]])

function_clf = clf.restriction([[-4, -4], [4, 4]], fun)
x = function_clf.random(20, class_id = "0")
print("Random points of the restricted classifier", x)


