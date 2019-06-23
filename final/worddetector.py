import cv2
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter, find_peaks


def crop_image(img, tol=0):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def gauss(kernlen=21):
    lim = kernlen // 2 + (kernlen % 2) / 2
    x = np.linspace(-lim, lim, kernlen + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    return kern1d


def detect_words(path_to_image):
    img = cv2.imread(path_to_image, 0)
    imgOrigin = img.copy()

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

    # Znajdowanie peaków
    roi, _ = find_peaks(rowsums_conv, threshold=0.4)
    imgOut = np.zeros_like(imgOrigin)
    line_number = 1
    indexes = []
    # Wykrywanie wyrazow
    for r in roi:
        if r - 15 > 0 and r + 45 < imgOrigin.shape[0]:
            th2_row = th2[r - 15:r + 45, :]
            col_sums = np.sum(th2_row, axis=0)
            col_sums_conv = np.convolve(col_sums, v, mode="valid")
            col_sums_conv = savgol_filter(col_sums_conv, 41, 3)

            starting_words, props = find_peaks(col_sums_conv, threshold=0.4, width=4, prominence=8, height=8)

            right_ips = [int(x) for x in props["right_ips"]]
            right_ips.sort()
            left_ips = [int(x) for x in props["left_ips"]]
            left_ips.sort()

            if len(left_ips) > 2:
                index = len(left_ips) - 1
                right_end = right_ips[index] + 30
                left_start = left_ips[index] - 5

                while index > 0:
                    index -= 1
                    new_right_end = right_ips[index] + 30
                    new_left_start = left_ips[index] - 5
                    if (left_start - new_right_end < 30 or right_end - left_start < 110) and (
                            right_end - new_left_start < 280):
                        # (gap is small or region is short) and region is not too long
                        left_start = new_left_start  # doklej lewy obszar
                    else:
                        break
                imgOut[r - 15:r + 45, left_start:right_end] = 127
                # cv2.imwrite('samples_second/' + path_to_image[8:-4] + '_' + str(r) + '.png',
                #             imgOrigin[r - 15:r + 45, left_start:right_end])
                indexes.append(imgOrigin[r - 15:r + 45, left_start:right_end])
    return indexes
