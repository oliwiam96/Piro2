import os

import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import load_model
from numpy import array

model = load_model('index_recognition_model.hdf5')


def recognize_single_index(img):
    img = np.array(img)
    img = cv2.GaussianBlur(img, (3, 3), 5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    #  ret3, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #  img = cv2.morphologyEx(img,cv2.MORPH_ERODE,(5,5),2)
    img = np.pad(img, ((0, 0), (0, 5)), 'constant')
    height, width = img.shape
    window_width = int(0.7 * height)
    move_window_step = int(0.14 * height)
    start = 0
    end = window_width
    consecutive = 0
    previous = -1
    index_text = ''

    while end <= width:
        th = img[:, start:end]
        size_image = cv2.resize(th, (34, 34))
        sample = size_image / 255
        sample = array(sample).reshape(34, 34, 1)
        sample = np.expand_dims(sample, axis=0)
        prediction = model.predict(sample)
        confidence = np.amax(prediction)
        predicted_class = np.argmax(prediction)
        # print(predicted_class)
        if confidence >= 0.5:
            # print("GOOD")
            current = predicted_class
            if current == previous:
                consecutive += 1
            else:
                consecutive = 0
            previous = current
            if consecutive == 1:
                index_text += str(predicted_class)
                consecutive = 0
                previous = -1
        start += move_window_step
        end += move_window_step
        # cv2.imshow('out', size_image)
        # cv2.waitKey(0)

    # tutaj sobie odkomentujcie jak chcecie mieć podgląd na bieżąco
    # print(index_text)
    # cv2.imshow('out', img)
    # cv2.waitKey(0)

    return ('', '', index_text)


def recognize_all():
    for file in os.listdir('../checkpoint1/samples_second'):
        recognize_single_index('../checkpoint1/samples_second/' + file)
