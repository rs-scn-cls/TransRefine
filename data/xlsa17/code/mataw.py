# import OS module
import os
import random
import csv
import scipy.io
import numpy as np

np.random.seed(2015)
# Get the list of all files and directories
path = "Provide_path_for_dataset" # Provide dataset path
dir_list = os.listdir(path)
ac_list = dir_list.copy()

# prints all files
np.random.shuffle(dir_list)
ac_list.sort()
images_lt = []
labels_lt = []
with open('UCM_img.csv','w') as csvfile: # change {dataset}_img.csv for other datasets
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(["Image Path"])
	for val in dir_list:
		#idx = ac_list.index(val)+1
		new_list = os.listdir(path+'/'+str(val))
		for vt in new_list:
			hel = str(path+'/'+str(val)+'/'+str(vt))
			#print(line)
			csvwriter.writerow([hel])
		
with open('UCM_label.csv','w') as csvfile:  # change {dataset}_label.csv for other datasets
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(["Label index"])
	for val in dir_list:
		idx = ac_list.index(val)+1
		new_list = os.listdir(path+'/'+str(val))
		for vt in new_list:
			#print(line)
			csvwriter.writerow([str(idx)])
