import os
from os import system
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
iris_OR = pd.read_csv("C:\\Users\\stefano\\PycharmProjects\\DataLab\\Cance\\data.csv")
iris_OR['diagnosis']=iris_OR['diagnosis'].map({'M':1,'B':0})
prediction_var=['concave points_worst', 'area_worst','texture_worst']

iris=iris_OR[prediction_var]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris_OR[prediction_var], iris_OR.diagnosis)
# tree.plot_tree(clf.fit(iris.data, iris.target))
# plt.show()

import graphviz

import collections
import pydotplus

dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=prediction_var,
                     class_names=['Benign','Malignant'],
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)

#graph = Source( tree.export_graphviz(dtreg, out_file=None, feature_names=X.columns))
graph.format = 'png'
graph.render('dtree_render_Falso',view=True)
graph = pydotplus.graph_from_dot_data(dot_data)

#graph


colors = ('orange', 'purple')
edges = collections.defaultdict(list)
x=graph.get_edge_list()
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
 #   edges[edge].sort()
    print(graph.get_node(str(edges[edge][0]))[0])
    print(str(edges[edge][0]))
    # for i in range(2):
    #     dest = graph.get_node(str(edges[edge][i]))[0]
    #     dest.set_fillcolor(colors[i])
   # if edge[0]>edge[1]
    #print(edge['obj_dict']['points'])

nodes = graph.get_node_list()

for node in nodes:
    if node.get_label():
        values = [int(ii) for ii in node.get_label().split('value = [')[1].split(']')[0].split(',')]
      #  values = [int(255 * v / sum(values)) for v in values]
      #  color = '#{:02x}{:02x}{:02x}'.format(values[0], values[1], values[2])
        if values[0]>values[1]:
          node.set_fillcolor(colors[0])
          #node.fontcolor(colors[0])
          node.attr(fontcolor='white')
        if values[0] < values[1]:
          node.set_fillcolor(colors[1])

graph.write_png('treXe.png')

graph
#tree.plot_tree(clf.fit(iris_OR[prediction_var], iris_OR.diagnosis))

