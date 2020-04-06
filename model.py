import os
import random
from time import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ReLU, \
    BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

random.seed(1001)

"""
Data loading
"""

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
print("The train dataset contains {} images".format(len(y_train)))

X_validation = np.load("X_validation.npy")
y_validation = np.load("y_validation.npy")

"""
Preparation of Training Data
"""

class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)
class_weights = dict(enumerate(class_weights))
print(class_weights)

y_train = to_categorical(y_train, dtype="int16")
with tf.device("/cpu:0"):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(500).batch(128).prefetch(1)


def train_preprocess(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x)
    x = tf.image.random_brightness(x, max_delta=.25)
    noise = tf.random.normal(shape=tf.shape(x), mean=0, stddev=1, dtype=tf.float32)
    x = tf.add(noise, x)
    x = x / 255
    return x, y


with tf.device("/cpu:0"):
    train_dataset = train_dataset.map(train_preprocess)
print(train_dataset)

image, label = next(iter(train_dataset))

for i in range(2):
    plt.plot(2, 1)
    plt.imshow(image[i])
    plt.show()

"""
Preparation of Validation Data
"""

y_validation = to_categorical(y_validation, dtype="int16")
validation_dataset = tf.data.Dataset.from_tensor_slices((X_validation, y_validation)).batch(128)


def validation_preprocess(x, y):
    x = x / 255
    return x, y


validation_dataset = validation_dataset.map(validation_preprocess)

print(len(X_train))
print(len(X_validation))
print(validation_dataset)

with tf.device("/gpu:0"):
    model = Sequential([Conv2D(64, 3, padding="same", input_shape=(64, 64, 3),
                               kernel_regularizer=l2(.0001)),
                        ReLU(),
                        BatchNormalization(),

                        Conv2D(64, 3, padding="same", kernel_regularizer=l2(.0001)),
                        ReLU(),
                        BatchNormalization(),
                        MaxPooling2D(),

                        Conv2D(128, 3, padding="same", kernel_regularizer=l2(.0001)),
                        ReLU(),
                        BatchNormalization(),

                        Conv2D(128, 3, padding="same", kernel_regularizer=l2(.0001)),
                        ReLU(),
                        BatchNormalization(),
                        MaxPooling2D(),

                        Conv2D(256, 3, padding="same", kernel_regularizer=l2(.0001)),
                        ReLU(),
                        BatchNormalization(),

                        Conv2D(256, 3, padding="same", kernel_regularizer=l2(.0001)),
                        ReLU(),
                        BatchNormalization(),
                        MaxPooling2D(),

                        Conv2D(512, 3, padding="same", kernel_regularizer=l2(.0001)),
                        ReLU(),
                        BatchNormalization(),

                        Conv2D(512, 3, padding="same", kernel_regularizer=l2(.0001)),
                        ReLU(),
                        BatchNormalization(),
                        MaxPooling2D(),

                        Flatten(),

                        Dense(64, kernel_regularizer=l2(.0001)),
                        ReLU(),
                        Dropout(.5),

                        Dense(10, activation="softmax")
                        ])
    model.summary()
    board = TensorBoard(log_dir="logs/{}".format(time()))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=["accuracy"])

    history = model.fit(train_dataset, verbose=2,
                        epochs=100, class_weight=class_weights, validation_data=validation_dataset,
                        callbacks=[board])

    train_score = model.evaluate(train_dataset, verbose=0)
    print('train loss, train acc:', train_score)

    validation_score = model.evaluate(validation_dataset, verbose=0)
    print('validation loss, validation acc:', validation_score)

    model.save("model.h5")

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
fig.savefig("accuracy_and_loss.jpg")



