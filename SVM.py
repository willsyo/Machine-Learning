import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

X = np.array([[0, 0, 0, 0],
             [0, 0, 1, 0],
             [1, 1, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 0, 1, 1],
             [0, 0, 0, 1],
             [1, 1, 0, 0]])

y = [0, 0, 0, 1, 1, 1, 1, 1]

clf = svm.SVC(C=20)

clf.fit(X,y)

print(clf.predict([[1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]]))
