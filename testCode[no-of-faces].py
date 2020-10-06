# THIS CODE IS FOR FINDING THE NUMBER OF FACES PRESENT IN AN IMAGE

from cv2 import cv2
import face_recognition

imgJohnny = face_recognition.load_image_file('assets/johnny-depp-and-robert.jpg')
imgJohnny = cv2.cvtColor(imgJohnny, cv2.COLOR_BGR2RGB)

johnnyLocation = face_recognition.face_locations(imgJohnny)

print(johnnyLocation)
print(len(johnnyLocation))
len = len(johnnyLocation)

for i in range(len):
    johnnyLocation1 = face_recognition.face_locations(imgJohnny)[i]
    cv2.rectangle(imgJohnny, (johnnyLocation1[3], johnnyLocation1[0]), (johnnyLocation1[1], johnnyLocation1[2]), (0, 255, 0), 1)

cv2.imshow('cssc', imgJohnny)
cv2.waitKey(0)
