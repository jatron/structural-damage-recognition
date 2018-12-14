# Setup
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
import os

checkpoint_path = "training/cp-freeze_layers_incept3_flatten_smalldatagen-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=10)

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


datagen.fit(X_training)

label_names = ['undamaged', 'damaged']

inception_v3 = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False)
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(inception_v3.layers):
    print(i, layer.name)
# we will freeze  alllayers and unfreeze the rest:
for layer in inception_v3.layers:
    layer.trainable = False

model = tf.keras.Sequential([
  inception_v3,
#  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(len(label_names), activation="softmax")])

# let's visualize layer names and layer indices
for i, layer in enumerate(model.layers):
    print(i, layer.name)

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.summary()

history = model.fit_generator(datagen.flow(X_training, y_training, batch_size=32), \
                              epochs=52, callbacks = [cp_callback], validation_data =(X_validation, y_validation), verbose=1)

pickle.dump(history.history, open("history_freeze_layers_incept3_flatten_smalldatagen.pkl", "wb"))
