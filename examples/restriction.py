import pyEDSD as edsd

clf = edsd.load("multi_3d.edsd")

restricted_clf = clf.restriction([1], [0.0])


options = {0 : {"colors" : "r"}, 
            1 : {"colors" : "g"},
            2 : {"colors" : "b"},
            3 : {"colors" : "c"}}

label_options = {0 : {"label" : "Test 1"}, 
            1 : {"label" : "Toto 2"},
            2 : {"label" : "Titi 3"},
            3 : {"label" : "Titi 3"}}


x = restricted_clf.random(class_id = 1)

restricted_clf.draw(plot_method = "classes", scatter = False, classes = [0, 1, 2], options = options, label_options = label_options)

clf.show()
