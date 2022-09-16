#ayarlar değişmediyse mavi cisim izler
import cv2
import numpy as np
from collections import deque  # tespit edilen obje merkezini depolamak icin

bufferSize = 16
pts = deque(maxlen=(bufferSize))

# mavi renk aralik HSV degerleri
blueLower = (84, 98, 0)
blueUpper = (179, 255, 255)

# caapture
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)
while True:
    success, imgOriginal = cap.read()
    if success:
        # blur
        blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0)

        # HSV format
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv image", hsv)

    # mavi rengi icin maske
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    cv2.imshow("mask image", mask)
    # maske etrafindaki gurultu sil
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("mask + erozyon ve genisleme", mask)

    # kontur
    (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        # en buyuk kontoru sec
        c = max(contours, key=cv2.contourArea)

        # konturu dikdÃ¶rtgene cevir
        rect = cv2.minAreaRect(c)

        ((x, y), (width, height), rotation) = rect
        print(x)
        # kutucuk hazirlama
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        # momentum
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # kontur cizdirme
        cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)

        # merkeze nokta cizme
        cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)
    # takip cizgisi
    pts.appendleft(center)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None: continue
        cv2.line(imgOriginal, pts[i - 1], pts[i], (0, 255, 0), 3)

    cv2.imshow("orjin", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()