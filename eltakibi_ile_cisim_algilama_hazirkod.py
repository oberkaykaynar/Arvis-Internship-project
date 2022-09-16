# **** LIBRARY SETUPS ****
# pip install mediapipe
# pip install opencv-python

import mediapipe as mp
import cv2
import numpy as np
import time

from collections import deque

# içerikler - contants
ml = 150
max_x, max_y = 250 + ml, 50
time_init = True
rad = 40


bufferSize = 16
pts = deque(maxlen=(bufferSize))

# mavi renk aralik HSV degerleri
blueLower = (75, 100, 100)
blueUpper = (130, 255, 255)
# Araçları seçme fonksiyonu - get tools function


def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True

    return False


hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# çizim araçları - drawing tools


mask = np.ones((480, 640)) * 255
mask = mask.astype('uint8')

cap = cv2.VideoCapture(0)

while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1) #görüntü çevirir

    if _:
        blurred = cv2.GaussianBlur(frm, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #cv2.imshow("hsv image", hsv)

    mask = cv2.inRange(hsv, blueLower, blueUpper)

    # maske etrafindaki gurultu sil
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        # en buyuk kontoru sec
        c = max(contours, key=cv2.contourArea)

        # konturu dikdÃ¶rtgene cevir
        rect = cv2.minAreaRect(c)

        ((cx, y), (width, height), rotation) = rect
        print("cisim:",cx)
        # kutucuk hazirlama
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        # momentum
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # kontur cizdirme
        cv2.drawContours(frm, [box], 0, (0, 255, 255), 2)

        # merkeze nokta cizme
        cv2.circle(frm, center, 5, (255, 0, 255), -1)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None: continue
        cv2.line(frm, pts[i - 1], pts[i], (0, 255, 0), 3)
    cv2.imshow("orjin", frm)  # takip


    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:

        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
            print("el:",x)

            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 1)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    time_init = True
                    rad = 40

            else:
                time_init = True
                rad = 40



    op = cv2.bitwise_and(frm, frm, mask=mask)
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]


    #cv2.imshow("mask + erozyon ve genisleme", mask)#takip
    #cv2.imshow("mask image", mask)
    cv2.imshow("Cizim Uygulamasi", frm)

    if cv2.waitKey(1) == 27:  # EXIT >>> ESC Key
        cv2.destroyAllWindows()
        cap.release()
        break