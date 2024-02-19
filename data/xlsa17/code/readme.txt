Data and code for "TRANSFREE: TRANSFORMER-AUGMENTED FEATURE REFINEMENT FOR ZERO-SHOT SCENE CLASSIFICATION IN REMOTE SENSING IMAGES"

Run the following codes one after the other.
--------------------------------------------------------------------------
		mataw.py
--------------------------------------------------------------------------
Initially, set the dataset path and run the mataw.py file.
This code will generate {dataset}_img/_label.csv files which contains: labels and image files.


--------------------------------------------------------------------------
mat_create.m
--------------------------------------------------------------------------
Code to create res101.mat file that contains image_files and labels.


--------------------------------------------------------------------------
dataset_create.py
--------------------------------------------------------------------------
Initially, set the dataset path and run the mataw.py file.
code to obtain att_splits.mat file, according to standard splits in remote sensing, we split the dataset. 
In order to obtain different splits, change the variable test_seen in this file according to standard splits (for eg., test_seen=16 is one of the standard splits in the UCM21 dataset (16/5)), and also change the random seed for every new split.

Then, place res101.mat and att_splits.mat files in TransFree/data/xlsa17/data/{dataset}/...


----------------------------------------------------------------------------
Data
----------------------------------------------------------------------------

resNet101.mat includes the following fields:
-labels: The label number of a class is its row number in allclasses.txt
-image_files: image sources  


att_splits.mat includes the following fields:
-att: columns correspond to class attribute vectors normalized to have unit l2 norm, following the classes order in {dataset}classes.txt 
-original_att: the original class attribute vectors without normalization
-trainval_loc: instances indexes of train+val set features (for only seen classes) in resnet101.mat
-test_seen_loc: instances indexes of test set features for seen classes
-test_unseen_loc: instances indexes of test set features for unseen classes
