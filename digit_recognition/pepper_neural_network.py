import tensorflow as tf
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
from random import randint


def pepper_and_salt(image):
    row, col = image.shape
    s_vs_p = 0.5
    amount = 0.009
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out


def lines_with_pepper(a):
    a = np.pad(a, (3, 3), 'constant')
    up_down = randint(-5, 5)
    a = np.roll(a, up_down, axis=0)
    a = np.roll(a, randint(-5, 5), axis=1)
    line_x = randint(0, 31)
    a[:, line_x] = 255
    a[up_down + 1, :] = 255
    a[up_down + 28, :] = 255
    a = pepper_and_salt(a)
    return a


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.array([lines_with_pepper(x) for x in x_train])
x_test = np.array([lines_with_pepper(x) for x in x_test])
x_train = x_train.reshape(x_train.shape[0], 34, 34, 1)
x_test = x_test.reshape(x_test.shape[0], 34, 34, 1)

input_shape = (34, 34, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.4))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_test, y_test))
model.save("pepper.hdf5")
