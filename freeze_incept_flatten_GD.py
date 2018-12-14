# Setup
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import pickle

# Pipe the dataset to a model
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_train_preprocessed = X_train / 128
X_train_preprocessed -= 1
print("X_train_preprocessed.shape:", X_train_preprocessed.shape)
print("y_train.shape:", y_train.shape)

label_names = ['undamaged', 'damaged']

inception_v3 = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False)
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(inception_v3.layers):
    print(i, layer.name)
# we will freeze the first 300 layers and unfreeze the rest:
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

model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

print("number of trainable variables:", len(model.trainable_variables))

model.summary()

history = model.fit(x=X_train_preprocessed, y=y_train, batch_size=32, epochs=15, validation_split=0.1)

pickle.dump(history.history, open("history_freeze_all_layers_incept3_add_Dense_Layer_GD_b32_lr0_01.pkl", "wb"))
