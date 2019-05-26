import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

img = cv2.imread('ocr1/img_19.jpg', 0)
imgOrigin = img.copy()


def crop_image(img, tol=0):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def gauss(kernlen=21):
    lim = kernlen // 2 + (kernlen % 2) / 2
    x = np.linspace(-lim, lim, kernlen + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    return kern1d


kernelH = np.ones((1, 7))
kernelV = np.ones((5, 1))
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))

# BLUR
img = cv2.GaussianBlur(img, (5, 5), 7)

# Threshold na podstawie któego znajdowanie są linie
th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Pozbycie się niektórych dziur w liniach poziomych/pionowych
thH = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 5)), iterations=1)
thV = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 3)), iterations=1)

# Threshold od którego odjęte zostaną znalezione linie
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 7)

# Wykrycie lini poziomych
sob = np.abs(cv2.Sobel(thH, -1, 0, 1, ksize=5))
sob = cv2.morphologyEx(sob, cv2.MORPH_OPEN, kernelH, iterations=3)
sob = cv2.morphologyEx(sob, cv2.MORPH_CLOSE, np.ones((3, 7)), iterations=1)
sob = np.roll(sob, 3, axis=0)

# Wykrycie lini pionowych
sob2 = np.abs(cv2.Sobel(thV, -1, 1, 0, ksize=5))
sob2 = cv2.morphologyEx(sob2, cv2.MORPH_OPEN, kernelV, iterations=3)
sob2 = cv2.morphologyEx(sob2, cv2.MORPH_CLOSE, np.ones((7, 3)), iterations=1)
sob2 = np.roll(sob2, 2, axis=1)

# Usunięcie lini z obrazka bianrnego
th2[sob > 0] = 0
th2[sob2 > 0] = 0

# Wypełnienie kształtów i pozbycie się szumu
th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, np.ones((7, 7)), iterations=5)
th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)

# Sumowanie wartości w kolumnie
rowsums = np.sum(th2, axis=1)
v = np.ones(40)
v[0:11] = 0.5
v[-11:] = 0.5

# Konwolucja z losowym filtrem żeby lepiej znaleść peaki w rozkładzie sum wierszy
rowsums_conv = np.convolve(rowsums, v, mode="valid")
rowsumsT = rowsums_conv.copy()

# Wygładzenie
rowsums_conv = savgol_filter(rowsums_conv, 41, 3)
plt.subplot(3, 1, 1)
plt.plot(rowsums)
plt.subplot(3, 1, 2)
plt.plot(rowsumsT)
plt.subplot(3, 1, 3)
plt.plot(rowsums_conv)
plt.show()

# Znajdowanie peaków
roi, _ = find_peaks(rowsums_conv, threshold=0.4)
print(roi)
for r in roi:
    print(r)
    if r - 10 > 0 and r + 40 < imgOrigin.shape[0]:
        cv2.imshow('pre', imgOrigin[r - 10:r + 40, :])
        cv2.waitKey(0)

cv2.imshow("pre", th2)
cv2.waitKey(0)
