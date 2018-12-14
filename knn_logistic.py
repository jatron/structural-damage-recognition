"""
================================
Digits Classification Exercise
================================

A tutorial exercise regarding the use of classification techniques on
the Digits dataset.

This exercise is used in the :ref:`clf_tut` part of the
:ref:`supervised_learning_tut` section of the
:ref:`stat_learn_tut_index`.

Changing code to look at damaged photos.
"""
print(__doc__)

from sklearn import datasets, neighbors, linear_model, metrics
import matplotlib.pyplot as plt
import numpy as np
import time

train_images = np.load('X_train.npy')
train_label = np.load('Y_train.npy')
test_images = np.load('X_test.npy')

"""
digits = datasets.load_digits()
X_digits = digits.data / digits.data.max()
y_digits = digits.target

n_samples = len(X_digits)

X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]
"""
start = time.time()
image_size=train_images.shape[1]*train_images.shape[2]*train_images.shape[3]
n_samples=train_images.shape[0]
#n_samples = 1000
ltrain_images = np.zeros((n_samples,image_size))
for i in range(n_samples):
    ltrain_images[i,:] = np.reshape(train_images[i,:], image_size) 

ltrain_images = ltrain_images / ltrain_images.max()
X_train = ltrain_images[:int(.9 * n_samples)]
y_train = train_label[:int(.9 * n_samples)]
#X_test = ltrain_images[int(.9 * n_samples):]
#y_test = train_label[int(.9 * n_samples):]
X_test = ltrain_images[int(.9 * n_samples):n_samples]
y_test = train_label[int(.9 * n_samples):n_samples]

start = time.time()
knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000,
                                           multi_class='multinomial')
end = time.time()
print('\nModel time (1000-images): {:.3f}s\n'.format(end-start))

start = time.time()

print("Train errors")
start = time.time()
knn_pred = knn.fit(X_train, y_train).predict(X_train)
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_train, y_train))
end = time.time()
print('\nKnn time (1000-images): {:.3f}s\n'.format(end-start))
start = time.time()
logistic_pred = logistic.fit(X_train, y_train).predict(X_train)
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_train, y_train))
end = time.time()
print('\nLogistic time (1000-images): {:.3f}s\n'.format(end-start))

print("Validation errors")
start = time.time()
knn_pred = knn.fit(X_train, y_train).predict(X_train)
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
end = time.time()
print('\nKnn time (1000-images): {:.3f}s\n'.format(end-start))
start = time.time()
logistic_pred = logistic.fit(X_train, y_train).predict(X_train)
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))
end = time.time()
print('\nLogistic time (1000-images): {:.3f}s\n'.format(end-start))

#knn_pred = knn.fit(X_train, y_train).predict(X_test)
#logistic_pred = logistic.fit(X_train, y_train).predict(X_test)
#print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
#print('LogisticRegression score: %f'
#      % logistic.fit(X_train, y_train).score(X_test, y_test))

#knn_prob = knn.fit(X_train, y_train).predict_prob(X_test)
#logistic_prob = logistic.fit(X_train, y_train).predict_prob(X_test)

knn_conf = metrics.confusion_matrix(y_test, knn_pred)
log_conf = metrics.confusion_matrix(y_test, logistic_pred)

#end = time.time()
#print('\nFit time (100-images): {:.3f}s\n'.format(end-start))
np.savez('knn_results', knn, knn_pred, knn_conf)
np.savez('knn_results', knn_pred, knn_conf)
np.savez('log_results', logistic, logistic_pred, log_conf)
