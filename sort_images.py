# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script file to save jpg images for Task #2 of the PHI Challenge
To save test images, first create a subfolder called "test"
To save training images, first create two subfolders called "damaged" and "undamaged"
All the images are named "image"

Use the flag train_sort to determine which you wish to sort (test or training)
Also need to use the flag save_flg to determine if you just wish to sort (or sort and save)
Finally use the show_flag to determine if you want to see the images
"""

import numpy as np
#import math 
#import matplotlib
#import scipy.misc
import imageio as image
from matplotlib import pyplot as plt
#from PIL import Image
from sklearn import datasets, neighbors, linear_model, metrics
import os

sort_type = 2   # 0 for no sort, 1 for sorting the training images, 2 for sorting the test images
save_flg=0    # For saving images into files
show_flg=0    # For plotting images
check_results=1    # for checking final results against labeled results
spt_flg=0    # For splitting images into validation and test sample sets

# Read in the data
train_images = np.load('../data/X_train.npy')
train_label = np.load('../data/Y_train.npy')
#test_images = np.load('../data/X_test.npy')
n_samples = len(train_images)

# Splits images into a percentage of training and validation samples (here a 90/10 split)
if (spt_flg==1):   
    X_train = train_images[:int(.9 * n_samples)]
    y_train = train_label[:int(.9 * n_samples)]

    # Split into validation and test sets
    X_test = train_images[int(.9 * n_samples):n_samples]
    y_test = train_label[int(.9 * n_samples):n_samples]
    test_ind = np.arange(int(.9 * n_samples),n_samples)
    
    # Keep the original indices with the array
    fy_test = np.column_stack((test_ind,y_test))    
    
    np.save('../X_test.npy', X_test)
    np.save('../y_test.npy', y_test)
    np.save('../fy_test.npy', fy_test)
else:
    X_train = train_images[:len(train_set)]
    y_train = train_label[:len(train_set)]
    y_test = None
    
    if (sort_type == 2):
    # Pick out the correct test images
        n_test = 1000
        X_test = train_images[len(train_set)-n_test:len(train_set)]
        y_test = train_label[len(train_set)-n_test:len(train_set)]
        test_ind = np.arange(len(train_set)-n_test,len(train_set))


# Save the training images
if (sort_type == 1):
    
    
    # Paths to the training image folders
    train_set = range(len(train_images))   # Number of training images to use
    #train_set = range(5000)   # Number of training images to use
    print("The number in the training set is "+str(len(train_set)))
    path_train = "../data/task2/train"+str(len(train_set))
    if (os.path.exists(path_train)): print(path_train+" exists") 
    else: os.mkdir(path_train)
    path_train_damaged = "../data/task2/train"+str(len(train_set))+"/damaged"
    if (os.path.exists(path_train_damaged)): print(path_train_damaged+" exists") 
    else: os.mkdir(path_train_damaged)
    path_train_undamaged = "../data/task2/train"+str(len(train_set))+"/undamaged"
    if (os.path.exists(path_train_undamaged)): print(path_train_undamaged+" exists") 
    else: os.mkdir(path_train_undamaged)

    damaged_ctr=0
    undamaged_ctr=0
    damaged_train_ind=[]
    undamaged_train_ind=[]
    for index in train_set: 
        img = X_train[index]
        if (y_train[index]==0):
            undamaged_train_ind.append(index)
            undamaged_ctr +=1
            if (save_flg):
                filename=os.path.join(path_train_undamaged,"t2_tr" + str(index)+ ".jpg")
                image.imwrite(filename,img)
        else:
            damaged_train_ind.append(index)
            damaged_ctr+=1
            if (save_flg):
                filename = os.path.join(path_train_damaged, "t2_tr" + str(index) + ".jpg")
                image.imwrite(filename, img)
    damaged_train_ind = np.array(damaged_train_ind)
    undamaged_train_ind = np.array(undamaged_train_ind)
    print("The number of damaged structures is "+str(damaged_ctr)+" and the percentage of damaged structures is "+str(damaged_ctr/len(train_set)))
    print("The number of undamaged structures is "+str(undamaged_ctr)+" and the percentage of undamaged structures is "+str(undamaged_ctr/len(train_set)))

# Save the test images
else:
    if (sort_type == 2):
        
        # Paths to the test folder
        path_test = "../data/task2/test4000" 
        path_test = "../data/task2/last1000"
        
        if (os.path.exists(path_test)): print(path_test+" exists") 
        else: os.mkdir(path_test)

        damaged_test_ctr=0
        undamaged_test_ctr=0
        for index in range(n_test): 
            img = X_test[index]
            filename=os.path.join(path_test,"tst" + str(test_ind[index])+ ".jpg")
            if save_flg:
                image.imwrite(filename, img)
                
            if (y_test[index]==0):
                undamaged_test_ctr +=1
            else:
                damaged_test_ctr +=1
        print("The number of damaged structures is "+str(damaged_test_ctr)+" and the percentage of damaged structures is "+str(damaged_test_ctr/n_test))
        print("The number of undamaged structures is "+str(undamaged_test_ctr)+" and the percentage of undamaged structures is "+str(undamaged_test_ctr/n_test))

# Plot the images
if (show_flg==1):
    images = train_images[undamaged_train_ind[0:36]]
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 6
    rows = 6
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        frame1= plt.imshow(images[i-1])
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
    finfig = plt.show()

if (check_results ==1):
    if (y_test[0]==None):
        print("y_test not defined")
    else:
        
        y_predm = np.loadtxt("../gcloud/try_results.txt", delimiter=", ")
        cm = metrics.confusion_matrix(y_test,y_predm[:,0])
        accuracy = sum(np.diag(cm))/np.sum(cm)

        if (0):
            y_predm = np.loadtxt("../gcloud/mobilenet_results_nov19_9pm.txt", delimiter=", ")
            cm = metrics.confusion_matrix(y_test,y_predm[:,0])
            accuracy = sum(np.diag(cm))/np.sum(cm)
            
            
            y_predi = np.loadtxt("../gcloud/incept_res_nov19_11pm.txt", delimiter=", ")
            ci = metrics.confusion_matrix(y_test,y_predi[:,0])
            accuracy = sum(np.diag(ci))/np.sum(ci)
