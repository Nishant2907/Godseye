# THIS IS A SHORTENED CODE
# WE ARE ALSO CHECKING THE TIME TAKEN BY THE CODE IF IMAGE IS NOT SHOWN

from cv2 import cv2
import face_recognition
import os
import time

# FOR CHECKING THE CPU TIME
startTimer = time.process_time()

# FUNCTION TO GET FACE LOCATION AND FACE ENCODINGS
def returnImageDetails(imagePath):
    image = face_recognition.load_image_file(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    location = face_recognition.face_locations(image)[0]
    encode = face_recognition.face_encodings(image)[0]
    return [location, encode]

imagePath = os.listdir('assets')
testImageDetails = returnImageDetails('testAssets/johnny-depp.jpg')
for i in range(len(imagePath)):
    imageDetails = returnImageDetails('assets/' + imagePath[i])
    result = []
    result = face_recognition.compare_faces([imageDetails[1]], testImageDetails[1])
    print(imagePath[i], result)

# FOR PRINTING THE TIME TAKEN TO EXECUTE THE CODE
print(time.process_time() - startTimer)
