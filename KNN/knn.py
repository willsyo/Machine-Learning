#!/usr/bin/env python
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):

	# Make sure k is greater than number of classes
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
	
	# Calculate euclidean distances from each point in each group
    distances = []
	# Groups
    for group in data:
		# Features within each group
        for features in data[group]:
			# calculate distance and append it to the array
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
			
    votes = [i[1] for i in sorted(distances)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
  
    return vote_result, confidence


df = pd.read_csv("breast_cancer_full.csv")
df.replace('?',-99999, inplace=True) 		
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()# Casting the data into float values
random.shuffle(full_data)					# Shuffle data

test_size = 0.4	
train_set = {2:[], 4:[]} # All benign points go into first list, and all melignants go into 2nd list
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))] # takes 60% of data for training set
test_data = full_data[-int(test_size*len(full_data)):]


for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1]) 

correct = 0
total = 0

# Perform KNN and test accuracy of algorithm
for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_nearest_neighbors(train_set, data, k=3)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/float(total))