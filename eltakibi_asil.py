import cv2
import mediapipe
import time

camera = cv2.VideoCapture(0)
mpHands = mediapipe.solutions.hands #elimizde 21 tane noktalar arasındaki bağlantıları çizdirmemizi sağlayacak
hands = mpHands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
#el objesi oluşturmuş oluruz.
# static_image_mode: resim şeklinde alacaksak True olmalı,Video olacaksa False olmalı,
#max_num_hands: en fazla kaç tane eli tanıyacak
#model_complexity= yalnızca 0 ya da 1 değerlerini alır
#min_detection_confidence= Tespitin başarılı sayılması için kişi tespit modelinden minimum güven değeri
# sayısal değer küçüldükçe el bulma oranı artıyor fakat videonun fps değeri düşüyor
#min_tracking_confidence=özellikle resimlerde işe yarar ve değer 0 a yaklaştıkça el tanıma artar
#bu parametreler girilmezse default değerler girilir
mpDraw = mediapipe.solutions.drawing_utils #elde ettiğimiz noktaları kamera üzerinde çizer
while camera.isOpened()==True:
    success, img = camera.read()
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hlms = hands.process(imgRGB) #görüntüyü rgb olarak işler
        #print(hlms.multi_hand_landmarks) #çıktı olarak listenin içinde 21 adet dict görüyoruz ve bunlarda eklem yerleri oluyor.
        # bu sayılar 0-1 arasındadır ve x,y yi kameranın weight,height ile çarparsak koordinat yerini bulmuş oluruz
        height, width, channel = img.shape
        #print(height,width)
        if hlms.multi_hand_landmarks: #none ifadesi dönmezse yani elimizi algılarsa buraya gir
            for handlandmarks in hlms.multi_hand_landmarks: #21 elemanı for ile tek tek gönderiyoruz
                mpDraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS) #landmarklar arası bağlantıları çiziyor
                for fingerNum, landmark in enumerate(handlandmarks.landmark):
                    positionX, positionY = int(landmark.x * width), int(landmark.y * height) #X ve Y koordinat yerlerini bulunuyor
                    print(positionX)
                    if fingerNum==8: #işaret parmağının ucu
                        cv2.circle(img,(positionX,positionY),10,(0,0,255),thickness=cv2.FILLED) #işaret parmağına daire şekli
        cv2.imshow("Camera", img)
    else: break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()