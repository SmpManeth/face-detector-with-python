from ast import While
import cv2
from random import randrange  # getting random colors

# load some pre trained data
tained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an Image to detect a face
# img = cv2.imread('rdj.jpg')

# capture Video from webcam
webcam = cv2.VideoCapture(0)


while True:
    succesful_frame_read, frame = webcam.read()
    # convert image to grey scale
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces and get the face cordonatres
    face_coordinates = tained_face_data.detectMultiScale(grayscale_image)
    
    for (x, y, w, h) in face_coordinates:
        # draw the rectangel
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,256, 0), 2)
        
    cv2.imshow('Detect Faces', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) == 81 or cv2.waitKey(1) == 113:
      break
  
webcam.release()
