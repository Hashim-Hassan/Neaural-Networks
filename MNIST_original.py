import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sklearn
sklearn.fetch('mldata')
from sklearn.datasets import fetch_mlDa
dataset = fetch_mldata('MNIST original')

X = dataset.data
y = dataset.target

some_digit = X[62302]
some_digit_image = some_digit_reshape(28, 28)

plt.imgshow(some_digit_image)
plt.show()

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X, y)

dtf.score(X, y)

dtf.predict(X[[17,2703, 13413, 56404, 62302], ])

from sklearn.tree import export_graphviz

export_graphviz(dtf, out_file="tree.dot")

import graphviz
with open("tree.dot") as f:
dot_graph = f.read()
graphviz.Source(dot_graph)



dataset = pd.read_csv('dataset/housing.csv')

import seaborn as sns
corr_mat = datset.corr()
sns.heatmap(corr_mat, annot = true)
pd.scatter_matrix(dataset)