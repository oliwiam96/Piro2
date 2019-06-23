import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# cv2.imshow('out3',x_train[4])
# cv2.waitKey(0)
# x_train = cv2.GaussianBlur(x_train, (1, 1), 5)
x_train = np.array([cv2.morphologyEx(x, cv2.MORPH_ERODE, np.ones((2, 2)), iterations=1) for x in x_train])
# cv2.imshow('out3',x_train[12])
# cv2.waitKey(0)
# x_test = cv2.GaussianBlur(x_test, (1, 1), 5)
x_test = np.array([cv2.morphologyEx(x, cv2.MORPH_ERODE, np.ones((2, 2)), iterations=1) for x in x_test])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

image_gen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # rescale=1./255,
    rotation_range=5,
    width_shift_range=.15,
    height_shift_range=.2, )

# training the image preprocessing
image_gen.fit(x_train, augment=True)
image_gen.fit(x_test, augment=True)

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
model.add(Dense(256, activation=tf.nn.relu))
model.add(Dropout(0.4))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# history = model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_test, y_test))
batch_size = 64
model.fit_generator(image_gen.flow(x_train, y_train, batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=40,
                    verbose=1, validation_data=(x_test, y_test))
model.save("digit_recognition_model.hdf5")
