import struct
from random import randint

import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator


def pepper_and_salt(image):
    row, col = image.shape
    s_vs_p = 0.5
    amount = 0.012
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


def load_idx(path):
    with open(path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        X = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        X = X.reshape((size, nrows, ncols))
    return X


def lines_with_pepper(a):
    a = np.pad(a, (3, 3), 'constant')
    up_down = randint(-5, 4)
    #  a = np.roll(a, up_down, axis=0)
    line_x = randint(1, 30)
    a[:, line_x - 1:line_x + 1] = 255
    a[up_down + 1, :] = 255
    a[up_down + 2, :] = 255
    a[up_down + 28, :] = 255
    a[up_down + 29, :] = 255
    a = pepper_and_salt(a)
    return a


image_gen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # rescale=1./255,
    zoom_range=0.2,
    shear_range=0.05,
    width_shift_range=.15,
    height_shift_range=.15, )

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# x_train =idx2numpy.convert_from_file('emnist/emnist-digits-train-images-idx3-ubyte')
# x_test = idx2numpy.convert_from_file('emnist/emnist-digits-test-images-idx3-ubyte')
# # y_train = load_idx('emnist/emnist-digits-train-labels-idx1-ubyte')
# # y_test = load_idx('emnist/emnist-digits-test-labels-idx1-ubyte')
# y_train = idx2numpy.convert_from_file('emnist/emnist-digits-train-labels-idx1-ubyte')
# y_test = idx2numpy.convert_from_file('emnist/emnist-digits-test-labels-idx1-ubyte')
# cv2.imshow('out',x_train[50])
# cv2.waitKey(0)
# print(y_train[50])
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
#
# cv2.imshow('out',x_train[40000])
# cv2.waitKey(0)
X = np.array_split(x_train, 3)
X_test = np.array_split(x_test, 3)

X[1] = np.array([lines_with_pepper(x) for x in X[1]])
X_test[1] = np.array([lines_with_pepper(x) for x in X_test[1]])

X[2] = np.array([lines_with_pepper(x) for x in X[2]])
X_test[2] = np.array([lines_with_pepper(x) for x in X_test[2]])

X.append(x_train)
X_test.append(x_test)

y_train2 = np.append(y_train, y_train)
y_test2 = np.append(y_test, y_test)

X.append(x_train)
X_test.append(x_test)

y_train = np.append(y_train2, y_train)
y_test = np.append(y_test2, y_test)

X[3] = np.array([lines_with_pepper(cv2.dilate(x, (3, 3), 1)) for x in X[3]])
X_test[3] = np.array([lines_with_pepper(cv2.dilate(x, (3, 3), 1)) for x in X_test[3]])
# cv2.imshow('out',X[3][5])
# cv2.waitKey(0)
X[4] = np.array([cv2.dilate(lines_with_pepper(x), (3, 3), 1) for x in X[4]])
X_test[4] = np.array([cv2.dilate(lines_with_pepper(x), (3, 3), 1) for x in X_test[4]])

X[0] = np.array([cv2.resize(x, (34, 34)) for x in X[0]])
X_test[0] = np.array([cv2.resize(x, (34, 34)) for x in X_test[0]])

x_train = np.array([item for sublist in X for item in sublist])
x_test = np.array([item for sublist in X_test for item in sublist])

# cv2.imshow('out',x_train[-1])
# cv2.waitKey(0)
x_train = x_train.reshape(x_train.shape[0], 34, 34, 1)
x_test = x_test.reshape(x_test.shape[0], 34, 34, 1)

input_shape = (34, 34, 1)

print(y_train[50])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(2, 2), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# history = model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_test, y_test))
batch_size = 128
model.fit_generator(image_gen.flow(x_train, y_train, batch_size),
                    steps_per_epoch=x_train.shape[0] * 2 // batch_size,
                    epochs=6,
                    verbose=1, validation_data=image_gen.flow(x_test, y_test, batch_size), shuffle=True,
                    validation_steps=x_test.shape[0] // batch_size)

model.save("pepper14.hdf5")
