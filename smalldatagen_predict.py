# Setup
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn import metrics

def create_model():
    inception_v3 = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False)
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(inception_v3.layers):
#    print(i, layer.name)
# we will freeze  alllayers and unfreeze the rest:
    for layer in inception_v3.layers:
        layer.trainable = False

    label_names = ['undamaged', 'damaged']
    model = tf.keras.Sequential([
        inception_v3,
#  tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(label_names), activation="softmax")])

# let's visualize layer names and layer indices
#for i, layer in enumerate(model.layers):
#    print(i, layer.name)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
    return model

checkpoint_path = "training/cp-freeze_layers_incept3_flatten_smalldatagen-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_model = "training/cp-freeze_layers_incept3_flatten_smalldatagen-0050.ckpt.data-00000-of-00001"

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

model = create_model()
model.load_weights(latest)

# Pipe the dataset to a model
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_train_preprocessed = X_train / 128
X_train_preprocessed -= 1
print("X_train_preprocessed.shape:", X_train_preprocessed.shape)
print("y_train.shape:", y_train.shape)

# Split into validation and test sets
n_samples = len(X_train_preprocessed)

X_training = X_train_preprocessed[:int(.9 * n_samples)]   # first 90% of the samples
y_training = y_train[:int(.9 * n_samples)]

X_validation = X_train_preprocessed[int(.9 * n_samples):n_samples]   # last 10% of the samples
y_validation = y_train[int(.9 * n_samples):n_samples]   
    
loss, acc = model.evaluate(X_training, y_training)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

y_ground_truth=y_training
X_data = X_training
y_pred_prob = model.predict(X_data, verbose = 1)
y_pred=np.argmax(y_pred_prob, axis=1)
np.savez('aug_results', y_ground_truth, y_pred_prob)
cm = metrics.confusion_matrix(y_ground_truth, y_pred)
accuracy = sum(np.diag(cm))/np.sum(cm)
print(cm)
print(accuracy)

false_pos = []
false_neg = []
for i in range(len(y_ground_truth)):
    if ((y_ground_truth[i]==0) & (y_pred[i]==1)):
        false_pos.append(i)
    if ((y_ground_truth[i]==1) & (y_pred[i]==0)):
        false_neg.append(i)
false_pos = np.array(false_pos)
false_neg = np.array(false_neg)
X_false_pos = X_data[false_pos,:,:,:]
X_false_neg = X_data[false_neg,:,:,:]
y_false_pos = y_ground_truth[false_pos]
y_false_neg = y_ground_truth[false_neg]

# Load data back in
#npzfile = np.load('./aug_results.npz')
# Index the result of interest
# npzfile.files (gives the variables)
# npzfile['x']  where x is the variable of interest

# Preprocess the data
#datagen = image.ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True,
#    shear_range=0.2,
#    fill_mode='nearest')

datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.05, fill_mode="nearest")

#datagen = image.ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,horizontal_flip=True)

datagen.fit(X_false_pos)

#model.summary()

history = model.fit_generator(datagen.flow(X_false_pos, y_false_pos, batch_size=32), \
                              epochs=10, callbacks = [cp_callback], validation_data =(X_validation, y_validation), verbose=1)

history = model.fit_generator(datagen.flow(X_false_pos, y_false_pos, batch_size=32), \
                              epochs=10, callbacks = [cp_callback], validation_data =(X_validation, y_validation), verbose=1)

#pickle.dump(history.history, open("history_freeze_layers_incept3_flatten_smalldatagen.pkl", "wb"))
