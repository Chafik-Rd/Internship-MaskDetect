from itertools import count
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from threading import Thread
from playsound import playsound
from tensorflow.python.keras.utils.generic_utils import has_arg

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]

#สีตัวกรอบที่วาดrandomใหม่ทุกครั้ง
COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))
#โหลดmodelจากแฟ้ม
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt","./MobileNetSSD/MobileNetSSD.caffemodel")

face_mask = ['Masked', 'No mask']
size = 224
model = tf.keras.models.load_model('/home/chafik/Internship/face_mask.model')
count = 1
speeking = False
t0 = time.time()
tstart = time.time()
result_ten_last = []
  
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5)
# For webcam input:
cap = cv2.VideoCapture(0)
with face_detection and hands:
  while cap.isOpened():
    success, image = cap.read()
    if success:
      (h,w) = image.shape[:2]
      #ทำpreprocessing
      blob = cv2.dnn.blobFromImage(image, 0.007843, (300,300), 127.5) #MobileNetSSD input (300,300)
      net.setInput(blob)
      detections = net.forward() #feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections

      for i in np.arange(0, detections.shape[2]):
        percent = detections[0,0,i,2]
        
        #กรองเอาเฉพาะค่าpercentที่สูงกว่า0.5 เพิ่มลดได้ตามต้องการ
        if percent > 0.75:
          class_index = int(detections[0,0,i,1])
          # print(class_index)
          box = detections[0,0,i,3:7]*np.array([w,h,w,h])
          (startX, startY, endX, endY) = box.astype("int")
          # print(COLORS)
          #ส่วนตกแต่งสามารถลองแก้กันได้ วาดกรอบและชื่อ
          label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
          cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[class_index], 2)
          cv2.rectangle(image, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
          y = startY - 15 if startY-15>15 else startY+15
          cv2.putText(image, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
          
          
    if not success:
      print("Ignoring empty camera frame.")
      continue   # If loading a video, use 'break' instead of 'continue'.
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # the BGR image to RGB. Because mediapipe use RGB
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False 
    results_face = face_detection.process(image)
    results_hands = hands.process(image)
    # print(results.detections)
    # print(results.detections)
    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results_face.detections:
      for detection in results_face.detections:
        location_face = detection.location_data.relative_bounding_box
        x = int(location_face.xmin * image.shape[1])
        y = int(location_face.ymin * image.shape[0])
        w = int(location_face.width * image.shape[1])
        h = int(location_face.height * image.shape[0])
        face_area = location_face.width * location_face.height
        # print(face_area)
        # print(results_face.detections)
        
        if x < 0 or y < 0:
            x = 0
            y = 0
        
        if face_area < 0.03:
            pass
        else:    
            crop_img = image[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (size, size))
            crop_img = np.reshape(crop_img, (1, size, size, 3)) / 255.0
            # print(crop_img)
            result = np.argmax(model.predict(crop_img))
            result_mask = np.argmax(model.predict(crop_img))
            result_ten_last.append(result_mask)
            if len(result_ten_last) > 10:
              result_ten_last.pop(0)
              
            
            if result_mask == 0:
                label = face_mask[0]
                color = (0, 255, 0)
                
                if results_hands.multi_hand_landmarks:
                  for hand_landmarks in results_hands.multi_hand_landmarks:
                    location_hands = hand_landmarks.landmark
                    x_hands = int((location_hands[4].x) * image.shape[1])
                    y_hands = int((location_hands[12].y) * image.shape[0])
                    w_hands = int((location_hands[20].x) * image.shape[1]) - int((location_hands[4].x) * image.shape[1])
                    h_hands = int((location_hands[0].y) * image.shape[0]) - int((location_hands[12].y) * image.shape[0])
                    hands_area = (location_hands[0].y - location_hands[12].y) * (location_hands[20].x - location_hands[4].x)
                    
                    # hands_area(-) is hand_left, hands_area(+) is hand_right
                    if hands_area < -0.040 or hands_area > 0.04: 
                      cv2.rectangle(image, (x_hands, y_hands), (x_hands+w_hands, y_hands+h_hands), (255, 0 , 0), 3)
                      # print(hands_area)
                      
                      # print(label_id_org)
                      if face_area > 0.1:
                        if time.time() - t0 > 2:
                          speeking = False
                        
                        if not speeking:
                          t0 = time.time()
                          t2 = Thread(target=playsound ,args=('speech1.mp3',))
                          t2.daemon = True
                          t2.start()
                          speeking = True
                          
                          img_item = "/home/chafik/Internship/data_person/{}.png".format(count)
                          frame_size = image[y-100:((y-100)+(h+150)), x-50:((x-50)+(w+100))]
                          cv2.imwrite(img_item, frame_size)
                          count = count+1
                          
                    # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
              label = face_mask[1]
              color = (0, 0, 255)
              
              if time.time() - t0 > 6:
                speeking = False
                
              if not speeking and np.mean(result_ten_last)> 0.5 and time.time() - tstart > 2 :
                t0 = time.time()
                t2 = Thread(target=playsound ,args=('speech.mp3',))
                t2.daemon = True
                t2.start()
                speeking = True
              
              # print(1)
              
            cv2.line(image,(0,240), (640,240), (0 ,0, 255), 3)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
            cv2.rectangle(image, (391, 168), (573, 422), (0 ,165, 255), 1)
            cv2.rectangle(image, (92, 209), (293, 409), (0 ,165, 255), 1)
            cv2.rectangle(image, (x, y - 60), (x+w, y), color, -1)
            cv2.putText(image, label, (x+ 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, "Hand", (391+ 10, 168 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,165, 255), 2)
            cv2.putText(image, "Face", (92+ 10, 209 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,165, 255), 2)
        
        # mp_drawing.draw_detection(image, detection)
        
    cv2.imshow('MediaPipe Face Detection',image)
    # print(time.time()-tstart)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
exit(1)