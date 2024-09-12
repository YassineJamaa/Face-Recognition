import cv2
import face_recognition
import dlib

img_will = face_recognition.load_image_file("Will-Ferrell.jpg")
img_will = cv2.cvtColor(img_will, cv2.COLOR_BGR2RGB)
img_will = cv2.resize(img_will, (300, 400))

# rectangle location
face_loc = face_recognition.face_locations(img_will)[0]
cv2.rectangle(img_will, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255,0,255), 2)
cv2.imshow("Will Furrell", img_will) 
cv2.waitKey(0)