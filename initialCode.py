# THIS IS THE FIRST ATTEMPT TO DETECT FACE IN AN IMAGE

from cv2 import cv2
import face_recognition

# FOR LOADING IMAGE FILE FROM "ASSETS" FOLDER
imgJohnny = face_recognition.load_image_file('assets/johnny-depp.jpg')

# ABOVE IMAGE WAS IN BGR COLOR FORMAT
# CONVERTING IT TO RGB COLOR FORMAT
imgJohnny = cv2.cvtColor(imgJohnny, cv2.COLOR_BGR2RGB)

# FINDING THE FACE IN THE IMAGE
# THIS GIVES US THE LOCATION OF FACE IN FORMAT --> TOP - RIGHT - BOTTOM - LEFT
johnnyLocation = face_recognition.face_locations(imgJohnny)[0]

# GETTING 128 MEASUREMENTS OF THE FACE
johnnyEncode = face_recognition.face_encodings(imgJohnny)[0]

# ADDING A GREEN COLOR RECTANGLE AROUND THE FACE
# HERE WE NEED TO PROVIDE FACE LOCATION IN FORMAT --> LEFT - TOP - RIGHT - BOTTOM
cv2.rectangle(imgJohnny, (johnnyLocation[3], johnnyLocation[0]), (johnnyLocation[1], johnnyLocation[2]), (0, 255, 0), 2)

# ADDING A NAME INSIDE THE IMAGE
cv2.putText(imgJohnny, "Johnny Depp", (johnnyLocation[3], johnnyLocation[2] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# REPEATING THE SAME STEP FOR A TEST IMAGE
imgJohnnyTest = face_recognition.load_image_file('assets/johnny-depp-test.jpg')
imgJohnnyTest = cv2.cvtColor(imgJohnnyTest, cv2.COLOR_BGR2RGB)
johnnyTestLocation = face_recognition.face_locations(imgJohnnyTest)[0]
johnnyTestEncode = face_recognition.face_encodings(imgJohnnyTest)[0]
cv2.rectangle(imgJohnnyTest, (johnnyTestLocation[3], johnnyTestLocation[0]), (johnnyTestLocation[1], johnnyTestLocation[2]), (0, 255, 0), 2)
cv2.putText(imgJohnnyTest, "Johnny Depp Test", (johnnyTestLocation[3], johnnyTestLocation[2] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# COMPARING BOTH FACES AND PRINTING THE RESULT
result = face_recognition.compare_faces([johnnyEncode],johnnyTestEncode)
print(result)

# SHOWING THE FINAL IMAGE WITH GREEN RECTANGLE
cv2.imshow('Johnny Depp', imgJohnny)
cv2.imshow('Johnny Depp Test', imgJohnnyTest)
cv2.waitKey(0)
