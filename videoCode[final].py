# THIS IS A FINAL VIDEO CODE

import face_recognition
from cv2 import cv2
import numpy as np
import math
import os

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

video_capture = cv2.VideoCapture(0)

ankit_image = face_recognition.load_image_file("assets/nishant_cropped.jpg")
ankit_face_encoding = face_recognition.face_encodings(ankit_image)[0]

  
known_face_encodings = []
known_face_names = []

allName = []
allEncode = []
allPath = os.listdir("testAssets")
for i in range(len(allPath)):
    allName.append(allPath[i].split(".")[0])
    temp = allName
    img = face_recognition.load_image_file("testAssets/" + allPath[i])
    allEncode.append(face_recognition.face_encodings(img))

known_face_names = allName
known_face_encodings = allEncode
print(known_face_names)
print(temp)


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #taking an frame from an the camera
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            #to check the accuracy of the face recognized
            print((face_distance_to_conf(face_distances))*100)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_ITALIC
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
