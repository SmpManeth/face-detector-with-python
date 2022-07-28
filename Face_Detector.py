import cv2

#load some pre trained data
tained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an Image to detect a face
img = cv2.imread('rdj.jpg')

#convert image to grey scale
grayscale_image = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates =tained_face_data.detectMultiScale(grayscale_image)


print(face_coordinates)


#show the Image
#cv2.imshow('Color Changer', grayscale_image)
cv2.waitKey()


print("Code Completed") 