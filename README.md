# structural-damage-recognition
sort_images.py   # This code takes the dataset and breaks it up into validation and test sets; plots damaged and undamaged images in 6x6 matrix; computes confusion matrix; plots false pos and false neg images; and plots augmented plots

retrain.py  # This code retrains the inceptionv3 model by adding a fully connected layer and softmax (Tensorflow).  It was modified from Tensorflow-for-Poets.

label_images.py  # This code does prediction using all files from a directory.  It was modified from Tensorflow-for-Poets (which did only one file at a time) and did not have any outputs of metrics.
