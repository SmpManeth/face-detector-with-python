import cv2

#load some pre trained data
tained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an Image to detect a face
img = cv2.imread('rdj.jpg')

#convert image to grey scale
grayscale_image = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)


#show the Image
cv2.imshow('Face Detector', grayscale_image)
cv2.waitKey()

print("Code Completed") 