# THIS IS THE FINAL CODE FOR VIDEO

from cv2 import cv2
import face_recognition
import os
import math
import numpy
from datetime import datetime

videoCapture = cv2.VideoCapture(0)

# FOR MARKING ATTENDANCE
def markAttendance(regNumber, name):
    f = open("Attendance.csv", "r+")
    data = f.readlines()
    regNumberList = []
    for line in data:
        entry = line.split(',')
        regNumberList.append(entry[0])
    if regNumber not in regNumberList:
        time = datetime.now()
        timeFormat = time.strftime('%H:%M:%S')
        f.writelines(f'\n{regNumber},{name},{timeFormat}')

# FOR CHECKING THE ACCURACY
def getAccuracy(faceDistance, faceMatchThreshold = 0.6):
    if faceDistance > faceMatchThreshold:
        range = (1.0 - faceMatchThreshold)
        linearValue = (1.0 - faceDistance) / (range * 2.0)
        return linearValue
    else:
        range = faceMatchThreshold
        linearValue = 1.0 - (faceDistance / (range * 2.0))
        return linearValue + ((1.0 - linearValue) * math.pow((linearValue - 0.5) * 2, 0.2))

# FOR GETTING PATH, NAME, REGISTRATION NO. AND ENCODINGS OF EACH PERSON
allPaths = os.listdir("videoTestAssets")
allNames = []
allRegNumbers = []
allEncodings = []
for index in range(len(allPaths)):
    allNames.append(allPaths[index].split(".")[0])
    allRegNumbers.append(allPaths[index].split(".")[1])
    image = face_recognition.load_image_file("videoTestAssets/" + allPaths[index])
    temp = face_recognition.face_encodings(image)[0]
    allEncodings.append(temp)

while True:
    ret, frame = videoCapture.read()

    frame = cv2.resize(frame, (0, 0), fx=2, fy=1.6)

    resizedFrame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    requiredFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)

    faceLocation = face_recognition.face_locations(requiredFrame)

    faceEncoding = face_recognition.face_encodings(requiredFrame, faceLocation)

    faceNames = []
    for encoding in faceEncoding:

        ismatched = face_recognition.compare_faces(allEncodings, encoding)
        matchedName = "Unknown"

        faceDistance = face_recognition.face_distance(allEncodings, encoding)

        if faceDistance[0] > faceDistance[1]: minimumFaceDistance = faceDistance[1]
        else: minimumFaceDistance = faceDistance[0]

        accuracy = getAccuracy(minimumFaceDistance)*100

        bestMatchIndex = numpy.argmin(faceDistance)

        #faceCoordinates = list(i*5 for i in faceLocation[0])

        if ismatched[bestMatchIndex] and accuracy > 80:
            matchedName = allNames[bestMatchIndex]
            markAttendance(allRegNumbers[bestMatchIndex], matchedName)
            #cv2.putText(frame, "%.2f"%accuracy + "%", (faceCoordinates[3] + 6, faceCoordinates[2]+ 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

        faceNames.append(matchedName)

    for (top, right, bottom, left), name in zip(faceLocation, faceNames):
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
        if(accuracy > 80):
            cv2.putText(frame, "%.2f"%accuracy + "%", (left + 6, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        

        #cv2.rectangle(frame, (faceCoordinates[3], faceCoordinates[0]), (faceCoordinates[1], faceCoordinates[2]), (0, 255, 0), 3)
        #cv2.putText(frame, matchedName, (faceCoordinates[3] + 6, faceCoordinates[2] - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
        
    cv2.imshow("Recording video", frame)
    cv2.waitKey(1)
