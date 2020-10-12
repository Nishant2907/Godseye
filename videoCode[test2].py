# THIS IS THE FIRST ATTEMPT WRITE CODE WHICH WILL BE USING WEBCAM

from cv2 import cv2
import face_recognition
import os
import math
import numpy

vid = cv2.VideoCapture(0)

# FOR CHECKING THE ACCURACY
def face_distance_to_conf(face_distance, face_match_threshold=0.4718):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

# NAME AND PATH OF EACH PERSON
allPath = os.listdir("testAssets")
allName = []
allEncode = []
for i in range(len(allPath)):
    allName.append(allPath[i].split(".")[0])
    img = face_recognition.load_image_file("testAssets/" + allPath[i])
#    allEncode.append(face_recognition.face_encodings(img))

for i in range(len(allPath)):
    for encode in allName:
        temp = encode
        img = face_recognition.load_image_file("testAssets/" + allPath[i])
        temp = face_recognition.face_encodings(img)
        allEncode.append(temp)

#nishant = face_recognition.load_image_file("assets/nishant_cropped.jpg")
#nishant = face_recognition.face_encodings(nishant)[0]

#hardik = face_recognition.load_image_file("assets/hardik.jpg")
#hardik = face_recognition.face_encodings(hardik)[0]

#allName = ["nishant", "hardik"]
#allEncode = [nishant, hardik]

X = True
while True:
    ret, frame = vid.read()

    resizeFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    checkFrame = resizeFrame[:, :, ::-1]

    
    if X:
        faceLocation = face_recognition.face_locations(checkFrame)

        faceEncode = face_recognition.face_encodings(checkFrame, faceLocation)

        #for i in range(len(allPath)):
        #    result = face_recognition.compare_faces([allEncode[i]], frame)
        #    if(result[0]):
        #        cv2.rectangle(frame, (100, 100), (100, 100), (0, 255, 0), 2)
        #    else:
        #        cv2.rectangle(frame, (100, 100), (100, 100), (0, 0, 255), 2)

        for i in faceEncode:
            match = face_recognition.compare_faces(allEncode, i)

            name = "Unknown"

            faceDistance = face_recognition.face_distance(allEncode, i)
            bestMatchIndex = numpy.argmin(faceDistance)

            if match[bestMatchIndex]:
                name = allName[bestMatchIndex]
                print(allName[bestMatchIndex])

    x = not X

    #cv2.rectangle(frame, (faceLocation[3], faceLocation[2]), (faceLocation[1], faceLocation[0]), (0, 255, 0), 2)
    cv2.imshow('Recording video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

