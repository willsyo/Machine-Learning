import numpy as np
import pylab as pl
import pandas as pd
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors, datasets
import random
from numpy.random import permutation

df = pd.read_csv("breast_cancer_full.csv")
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

random_indices = permutation(df.index)
max_length = math.floor(len(df)/3)

test = df.loc[random_indices[1:max_length]]
train = df.loc[random_indices[max_length:]]

x_col = ['Thickness','Size','Shape','Adhesion','Epithelial','Nuclei','Chromatin','Nucleoli','Mitosis']
y_col = ['Class']

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(train[x_col],train[y_col])
predictions = knn.predict(test[x_col])

actual_set = test[y_col]

actual = np.asarray(actual_set)

correct = 0
total = 0

for i in range(len(predictions)):
    if(predictions[i] == actual[i]):
        correct += 1
    total += 1

print('Accuracy: ', correct/float(total))
