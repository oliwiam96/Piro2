from functools import reduce

import cv2
import numpy as np

img = cv2.imread('ocr1/img_17.jpg')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel1 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((3, 3), np.uint8)
kernel = np.zeros((5,5), np.uint8)
kernel[2,:]=1
#kernel[:,3]=1

# Można użyć do wykrywania samych literek i cyfr
#edges = cv2.Canny(imgGrey,220,300,apertureSize = 3)

blur = cv2.GaussianBlur(imgGrey, (5, 5), 0)
edges = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
#can = cv2.erode(can, kernel, iterations=1)
ero = cv2.dilate(edges, kernel, iterations=1)
dil = cv2.erode(ero, kernel, iterations=2)


# Wykrywanie linii
lines = cv2.HoughLinesP(dil,1,np.pi/180,80,minLineLength=400,maxLineGap=20)

# Pozostawienie tylko linii poziomych
lines = list(filter(lambda l: abs(l[0][1]-l[0][3]) < 20,lines))
linesSorted = sorted(lines, key=lambda l:l[0][1])


l1 = list(map(lambda l: l[0][1],linesSorted[1:]))
l2 = list(map(lambda l: l[0][1],linesSorted[:-1]))
l3 = list(zip(l1,l2))

# Odległości w pionie pomiędzy liniami
ldiff = list(map(lambda l: l[0]-l[1],l3))

# Pozostawienie linii, które są oddalone o więcej niż 10 px
linesFiltered = []
dist_between_lines = 0
for i,d in enumerate(ldiff):
    if d>20:
        dist_between_lines+=d
        #print(d, linesSorted[i])
        linesFiltered.append(linesSorted[i])

#dist_between_lines= int(dist_between_lines/len(linesSorted))
dist_between_lines =  50

# for line in linesFiltered:
#     for x1,y1,x2,y2 in line:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


# Wykrywanie lini 2
# lines = cv2.HoughLines(edges,1,np.pi/90,300)
#
#
# for line in lines:
#     for rho, theta in line:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


cv2.imshow("a",img)
        #cv2.imwrite("a.jpg",img)
cv2.waitKey(0)

for line in linesFiltered:
    for x1,y1,x2,y2 in line:
        # Obszar obrazka 65px powyżej linii oraz 15px poniżej
        roi_from = max(0,y1-dist_between_lines-15)
        roi = img[roi_from:y1+15,:]

        # To jeszcze nie działa
        edges_roi = cv2.Canny(img[roi_from:y1+15,:],220,300,apertureSize = 3)
        im2, contours, hierarchy = cv2.findContours(edges_roi, 1, 2)
        edges_roi = cv2.dilate(edges_roi,kernel2, iterations=2)
        if len(contours)>0:
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("a",roi)
        #cv2.imwrite("a.jpg",img)
        cv2.waitKey(0)
