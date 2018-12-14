# structural-damage-recognition
sort_images.py   # This code takes the dataset and breaks it up into validation and test sets; plots damaged and undamaged images in 6x6 matrix; computes confusion matrix; plots false pos and false neg images; and plots augmented plots

retrain.py  # This code retrains the inceptionv3 model by adding a fully connected layer and softmax (Tensorflow).  It was modified from Tensorflow-for-Poets.

label_images.py  # This code does prediction using all files from a directory.  It was modified from Tensorflow-for-Poets (which did only one file at a time) and did not have any outputs of metrics.

knn_logistic.py  # This code does k-nearest neighbors and logistic regression.  It calls sklearn functions.  

svm_classification.py # This code does SVM on the data set.  It calls sklearn functions

freeze_first_300.py  # This code freeze the first 300 layers of the inceptionv3 network and trains the rest (300 to 312)

freeze_incept_avgpool.py  # This code freezes all the layers of the inceptionv3 network, remove top layer, adds an avgpool layer, then fc layer, then softmax

freeze_incept_dropout_between_two_dense.py  # Freeze all layers of inceptionv3, remove top layer, then add dense and dropout layers to reduce overfit

freeze_incept_dropout.py # Freeze all layers of inceptionv3, remove top layer, then add dropout layers to reduce overfit

freeze_incept_flatten_GD.py # compare Adam Prop with Gradient Descent

freeze_flatten_reg.py  # compare L2 regularization with dropout

freeze_incept_sigmoid.py  # replace softmax by sigmoid, since we only have two classes

freeze_incept_flatten_TFP.py  # freeze all layers of inceptionv3, keep the top layer, then add FC and softmax

freeze_incept_flatten.py  # freeze all layers of inceptionv3, remove top layer, then add dense and FC and softmax

freeze_inceptv3_flatten_datagen.py # augment the data with flip, zoom, and fill pixels (valid) and use different images for each iteration

freeze_inceptv3_flatten_smalldatagen.py # augment the data with flip, width_shift, and height_shift images (many not valid) and use different images for each iteration.  also enables saving of intermediate model checkpoints.

smalldatagen_predict.py # Enable saving and loading of model checkpoints and then perform prediction, accuracy, and confusion matrices

freeze_incept_twice.py # Train twice with the data; first pass for the new layer only; second pass, more layers with low learning rate.
