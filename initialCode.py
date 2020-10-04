from cv2 import cv2
import face_recognition

# FOR LOADING IMAGE FILE FROM "ASSETS" FOLDER
imgJohnny = face_recognition.load_image_file('assets/johnny-depp.jpg')

# ABOVE IMAGE WAS IN BGR COLOR FORMAT
# CONVERTING IT TO RGB COLOR FORMAT
imgJohnny= cv2.cvtColor(imgJohnny, cv2.COLOR_BGR2RGB)

# FINDING THE FACE IN THE IMAGE
johnnyLocation = face_recognition.face_locations(imgJohnny)[0]

# GETTING 128 MEASUREMENTS OF THE FACE
johnnyEncode = face_recognition.face_encodings(imgJohnny)[0]

# ADDING A GREEN COLOR RECTANGLE AROUND THE FACE
cv2.rectangle(imgJohnny, (johnnyLocation[3], johnnyLocation[0]), (johnnyLocation[1], johnnyLocation[2]), (0, 255, 0), 2)

# ADDING A NAME INSIDE THE IMAGE
cv2.putText(imgJohnny, "Johnny Depp", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# REPEATING THE SAME STEP FOR A TEST IMAGE
imgJohnnytest = face_recognition.load_image_file('assets/johnny-depp-test.jpg')
imgJohnnytest = cv2.cvtColor(imgJohnnytest, cv2.COLOR_BGR2RGB)
johnnytestLocation = face_recognition.face_locations(imgJohnnytest)[0]
johnnytestEncode = face_recognition.face_encodings(imgJohnnytest)[0]

# COMPARING BOTH FACES AND PRINTING THE RESULT
result = face_recognition.compare_faces([johnnyEncode],johnnytestEncode)
print(result)

cv2.rectangle(imgJohnnytest, (johnnytestLocation[3], johnnytestLocation[0]), (johnnytestLocation[1], johnnytestLocation[2]), (0, 255, 0), 2)
cv2.putText(imgJohnny, "Johnny Depp Test", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# SHOWING THE FINAL IMAGE WITH GREEN RECTANGLE
cv2.imshow('Johnny Depp', imgJohnny)
cv2.imshow('Johnny Depp Test', imgJohnnytest)
cv2.waitKey(0)


