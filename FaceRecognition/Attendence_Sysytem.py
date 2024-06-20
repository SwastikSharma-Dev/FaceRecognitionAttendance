import face_recognition
import cv2
import os
import numpy as np
import csv 
from datetime import datetime

video_capture=cv2.VideoCapture(0) #For Webcam
known_faces_names=[]
known_faces_encodings=[]
DIR=r'C:\Users\varun\OneDrive\Desktop\ELC\photos'
for i in os.listdir(DIR):
    a=i[:len(i)-5]
    known_faces_names.append(a)
    path=os.path.join(DIR,i)
    the_image=face_recognition.load_image_file(path)
    the_encoding=face_recognition.face_encodings(the_image)[0]
    known_faces_encodings.append(the_encoding)

students=known_faces_names.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True

now=datetime.now()
current_date=now.strftime("%d-%m-%Y")

f=open(current_date+'.csv','w+',newline='')
lnwriter=csv.writer(f)

while True:
    _, frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    if True:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_faces_encodings,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_faces_encodings,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_faces_names[best_match_index]
            
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    current_time=now.strftime("%H:%M:%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()