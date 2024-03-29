import cv2
import numpy as np
from numpy import array
from keras.models import load_model

model = load_model('pepper.hdf5')
img = cv2.imread('test3.png', 0)
height, width = img.shape
window_width = int(0.6 * height)
move_window_step = 8
start = 0
end = window_width
consecutive = 0
current = 0
previous = 0
index_text = ''

# img2 = cv2.GaussianBlur(img, (5, 5), 5)
img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 9)
# img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, np.ones((2, 2)), iterations=1)
# img2 = cv2.dilate(img2, np.ones((3, 3),np.uint8), iterations = 1)

cv2.imshow('out', img2)
cv2.waitKey(0)
while end <= width:
    current_window = img[:, start:end]
    # th = cv2.GaussianBlur(current_window, (4, 4), 4)
    th = cv2.adaptiveThreshold(current_window, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    # th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2, 2)), iterations=1)
    # th = cv2.dilate(th, np.ones((3, 3),np.uint8), iterations = 1)
    # th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((1,1)), iterations=3)
    # th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=1)
    size_image = cv2.resize(th, (34, 34))
    sample = size_image / 255
    sample = array(sample).reshape(34, 34, 1)
    sample = np.expand_dims(sample, axis=0)
    prediction = model.predict(sample)
    confidence = np.amax(prediction)
    predicted_class = np.argmax(prediction)
    print(predicted_class)
    if confidence >= 0.80:
        print("GOOD")
        print("Initial prediction: ", confidence, predicted_class)
        current = predicted_class
        if current == previous:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive == 2:
            index_text += str(predicted_class)
            consecutive = 0
            previous = -1
        previous = current
    else:
        print("BAD")
    cv2.imshow('out', size_image)
    cv2.waitKey(0)
    start += move_window_step
    end += move_window_step

print(index_text)
