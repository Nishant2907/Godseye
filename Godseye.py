from cv2 import cv2
import face_recognition

imgJohnny = face_recognition.load_image_file('assets/Johnny-Depp.jpg')
imgJohnny= cv2.cvtColor(imgJohnny, cv2.COLOR_BGR2RGB)

imgJohnnytest = face_recognition.load_image_file('assets/johnydepp-test.jpg')
imgJohnnytest = cv2.cvtColor(imgJohnnytest, cv2.COLOR_BGR2RGB)

johnnyLocation = face_recognition.face_locations(imgJohnny)[0]
johnnyEncode = face_recognition.face_encodings(imgJohnny)[0]
cv2.rectangle(imgJohnny, (johnnyLocation[3], johnnyLocation[0]), (johnnyLocation[1], johnnyLocation[2]), (0, 255, 0), 2)

johnnytestLocation = face_recognition.face_locations(imgJohnnytest)[0]
johnnytestEncode = face_recognition.face_encodings(imgJohnnytest)[0]
cv2.rectangle(imgJohnnytest, (johnnytestLocation[3], johnnytestLocation[0]), (johnnytestLocation[1], johnnytestLocation[2]), (255, 0, 0), 2)

cv2.imshow('Johnny Depp', imgJohnny)
cv2.imshow('Johnny Depp Test', imgJohnnytest)
cv2.waitKey(0)

result = face_recognition.compare_faces([johnnyEncode],johnnytestEncode)
print(result)