# THIS FILE IS FOR CHECKING IF THE CODE RUNS WHEN THE TEST IMAGE WILL HAVE MORE THAN ONE FACE

from cv2 import cv2
import face_recognition

# 1ST IMAGE
imgJohnny = face_recognition.load_image_file('assets/johnny-depp.jpg')
imgJohnny= cv2.cvtColor(imgJohnny, cv2.COLOR_BGR2RGB)
johnnyLocation = face_recognition.face_locations(imgJohnny)[0]
johnnyEncode = face_recognition.face_encodings(imgJohnny)[0]

# 2ND IMAGE
imgJohnnytest = face_recognition.load_image_file('testAssets/johnny-depp-and-robert.jpg')
imgJohnnytest = cv2.cvtColor(imgJohnnytest, cv2.COLOR_BGR2RGB)

faces = face_recognition.face_locations(imgJohnnytest)
len = len(faces)

for i in range(len):
    johnnytestLocation = face_recognition.face_locations(imgJohnnytest)[i]
    johnnytestEncode = face_recognition.face_encodings(imgJohnnytest)[i]
    result = face_recognition.compare_faces([johnnyEncode],johnnytestEncode)

    if(result[0]):
        cv2.rectangle(imgJohnny, (johnnyLocation[3], johnnyLocation[0]), (johnnyLocation[1], johnnyLocation[2]), (0, 255, 0), 2)
        cv2.rectangle(imgJohnnytest, (johnnytestLocation[3], johnnytestLocation[0]), (johnnytestLocation[1], johnnytestLocation[2]), (0, 255, 0), 2)
    else:
        cv2.rectangle(imgJohnny, (johnnyLocation[3], johnnyLocation[0]), (johnnyLocation[1], johnnyLocation[2]), (0, 255, 0), 2)
        cv2.rectangle(imgJohnnytest, (johnnytestLocation[3], johnnytestLocation[0]), (johnnytestLocation[1], johnnytestLocation[2]), (0, 0, 255), 2)


# SHOWING THE FINAL IMAGE WITH GREEN RECTANGLE
cv2.imshow('Johnny Depp', imgJohnny)
cv2.imshow('Johnny Depp Test', imgJohnnytest)
cv2.waitKey(0)
