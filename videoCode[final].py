# THIS IS THE FINAL CODE FOR VIDEO

from cv2 import cv2
import face_recognition
import os
import math
import numpy
from datetime import datetime

vid = cv2.VideoCapture(0)

# FOR MARKING ATTENDANCE
def markAttendance(regNo, name):
    f = open("Attendance.csv", "r+")
    data = f.readlines()
    regList = []
    for line in data:
        entry = line.split(',')
        regList.append(entry[0])
    if regNo not in regList:
        time = datetime.now()
        dtstring = time.strftime('%H:%M:%S')
        f.writelines(f'\n{regNo},{name},{dtstring}')

# FOR CHECKING THE ACCURACY
def accuracy(face_distance, face_match_threshold = 0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

# NAME AND PATH OF EACH PERSON
allPath = os.listdir("videoTestAssets")
allName = []
allReg = []
allEncode = []
for i in range(len(allPath)):
    allName.append(allPath[i].split(".")[0])
    allReg.append(allPath[i].split(".")[1])
    img = face_recognition.load_image_file("videoTestAssets/" + allPath[i])
    temp = face_recognition.face_encodings(img)[0]
    allEncode.append(temp)

while True:
    ret, frame = vid.read()

    resizeFrame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    checkFrame = cv2.cvtColor(resizeFrame, cv2.COLOR_BGR2RGB)

    faceLocation = face_recognition.face_locations(checkFrame)

    faceEncode = face_recognition.face_encodings(checkFrame, faceLocation)

    faceNames = []
    for i in faceEncode:

        match = face_recognition.compare_faces(allEncode, i)
        name = "Unknown"

        faceDistance = face_recognition.face_distance(allEncode, i)

        if faceDistance[0] > faceDistance[1]: minValue = faceDistance[1]
        else: minValue = faceDistance[0]

        accurate = accuracy(minValue)*100
        print(accurate)

        bestMatchIndex = numpy.argmin(faceDistance)

        if match[bestMatchIndex] and accurate>80:
            name = allName[bestMatchIndex]
            #markAttendance(name)
            markAttendance(allReg[bestMatchIndex], name)
         
        faceNames.append(name)
        print(name)
        
    for (top, right, bottom, left), name in zip(faceLocation, faceNames):
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('Recording video', frame)
    cv2.waitKey(1)