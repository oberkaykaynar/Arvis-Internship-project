#ekrana birer saniye aralıklarla rastgele renkler çıkıyor
import cv2
import numpy as np
import random
while True:
    random1=random.randint(300,800)
    random2=random.randint(300,800)
    randomcolor = np.uint8(np.random.random((1, 1, 3)) * 255)
    randomcolor=cv2.resize(randomcolor,(random1,random2))
    print("r",randomcolor[0][0])
    cv2.imshow("white",randomcolor)
    cv2.waitKey(1000)
cv2.destroyAllWindows()