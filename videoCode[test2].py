# THIS IS THE FIRST ATTEMPT WRITE CODE WHICH WILL BE USING WEBCAM

from cv2 import cv2
import face_recognition
import os

# NAME AND PATH OF EACH PERSON
allPath = os.listdir("testAssets")
allName = []
allEncode = []
for i in range(len(allPath)):
    allName.append(allPath[i].split(".")[0])
    img = face_recognition.load_image_file("testAssets/" + allPath[i])
    allEncode.append(face_recognition.face_encodings(img))


vid = cv2.VideoCapture()

while True:
    frame = vid.read()

    #frame = cv2.resize(frame,(0,0),None,0.25,0.25)

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    facelocation = face_recognition.face_locations(frameRGB)

    faceencode = face_recognition.face_encodings(frameRGB)

    for i in range(len(allPath)):
        result = face_recognition.compare_faces([allEncode[i]], frameRGB)
        if(result[0]):
            cv2.rectangle(frameRGB, (100, 100), (100, 100), (0, 255, 0), 2)
        else:
            cv2.rectangle(frameRGB, (100, 100), (100, 100), (0, 0, 255), 2)
    

