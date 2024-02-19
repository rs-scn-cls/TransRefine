# import OS module
import os
import random
import csv
import scipy.io
import numpy as np
from sklearn import preprocessing


vals = []
i = 0
with open('UCM_Dataset.csv','r') as csvfile: # change {dataset}_Dataset.csv for other datasets
  csvFile = csv.reader(csvfile)
  for lines in csvFile:
    if i != 0:
      vals.append(lines[1:])
    else:
      i = 1
      
X = np.asarray(vals, dtype=np.float64)
X_normalized = preprocessing.normalize(X, norm='l2')
X = np.transpose(X)
X_normalized = np.transpose(X_normalized)

mat = scipy.io.loadmat('res101.mat')

path = "Provide_path_for_the_dataset" # please provide the dataset path

dir_list = os.listdir(path)

labels = [*range(1,len(dir_list) + 1)]

label_map = []
label_map.append([])
for i in labels:
	label_map.append([])
for val in range(len(mat['labels'])):
	label_map[int(mat['labels'][val])].append(val+1)

#split
np.random.seed(2014)	#2014 2015 2016 2017 2018 2019
np.random.shuffle(labels)
test_seen = 16   #13 10 7 for UCM, 25, 20, 15, 10 for AID, 35, 30, 25, 20 for NWPU, 15, 12, 9, 6 for RSD datasets
test_unseen = len(dir_list) - test_seen
test_unseen = labels[test_seen:]
test_seen = labels[0:test_seen]

#train-test split
train = 0.8
test = 1 - train

A = []			#Train
B = []			#Test_Seen
for i in test_seen:
	A.extend(label_map[i][0:int(train*len(label_map[i]))])
	B.extend(label_map[i][int(train*len(label_map[i])):])
		
C = []
for i in test_unseen:
	C.extend(label_map[i])

np.random.shuffle(A)
np.random.shuffle(B)
np.random.shuffle(C)

A = np.reshape(A, (-1, 1))
B = np.reshape(B, (-1, 1))
C = np.reshape(C, (-1, 1))

obj_arr = np.zeros((5,), dtype=np.object_)
obj_arr[0] = A
obj_arr[1] = B
obj_arr[2] = C
obj_arr[3] = X
obj_arr[4] = X_normalized


scipy.io.savemat('att_splits.mat', mdict={'trainval_loc': obj_arr[0], 'test_seen_loc': obj_arr[1], 'test_unseen_loc': obj_arr[2], 'original_att': obj_arr[3], 'att': obj_arr[4]})
