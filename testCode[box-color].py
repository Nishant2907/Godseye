# THIS CODE CHANGES THE COLOR OF THE RECTAGLE AROUND THE FACE
# RED IF FACES DOESN'T MATCH
# GREEN IF THE FCES MATCH

from cv2 import cv2
import face_recognition

# FOR LOADING IMAGE FILE FROM "ASSETS" FOLDER
imgJohnny = face_recognition.load_image_file('assets/johnny-depp.jpg')

# ABOVE IMAGE WAS IN BGR COLOR FORMAT
# CONVERTING IT TO RGB COLOR FORMAT
imgJohnny= cv2.cvtColor(imgJohnny, cv2.COLOR_BGR2RGB)

# FINDING THE FACE IN THE IMAGE
# THIS GIVES US THE LOCATION OF FACE IN FORMAT --> TOP - RIGHT - BOTTOM - LEFT
johnnyLocation = face_recognition.face_locations(imgJohnny)[0]

# GETTING 128 MEASUREMENTS OF THE FACE
johnnyEncode = face_recognition.face_encodings(imgJohnny)[0]

# REPEATING THE SAME STEP FOR A TEST IMAGE
imgJohnnytest = face_recognition.load_image_file('assets/hardik.jpg')
imgJohnnytest = cv2.cvtColor(imgJohnnytest, cv2.COLOR_BGR2RGB)
johnnytestLocation = face_recognition.face_locations(imgJohnnytest)[0]
johnnytestEncode = face_recognition.face_encodings(imgJohnnytest)[0]

# COMPARING BOTH FACES AND PRINTING THE RESULT
result = face_recognition.compare_faces([johnnyEncode],johnnytestEncode)
print(result)

# IF FACES MATCH --> BOTH IMAGES WILL HAVE GREEN RECTANGLE
# ELSE TEST IMAGE WILL HAVE RED RECTANGLE
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
