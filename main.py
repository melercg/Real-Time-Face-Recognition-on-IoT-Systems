import os
import face_recognition
import cv2
import time

from utils.camera import Colors
from utils.camera import CameraUtils
from utils.dataset import Dataset

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SCALE_X = 0.25
SCALE_Y = 0.25
camera_utils = CameraUtils()




dataset_driver = Dataset(images_folder_path=BASE_DIR+'/Faces')
dataset = dataset_driver.DeSerializeImages(BASE_DIR+'/encodings/version_1.pickle')


capture = cv2.VideoCapture(camera_utils.Capture_Src)
unknown_face_locations = []
unknown_face_encodings = []
new_frame_time  = 0
prev_frame_time = 0
while True:
    font =cv2.FONT_HERSHEY_PLAIN

    unknown_face_names = []
    success, frame = capture.read()
    if not success:
        raise Exception("Camera connection is a lost!")
    new_frame_time = time.time() 
  
    
    
  
    resized_frame = cv2.resize(frame,(0,0),fx=camera_utils.Scale_X,fy=camera_utils.Scale_Y, interpolation=cv2.INTER_AREA)
    resized_rgb_frame = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2RGB)

    unknown_face_locations = face_recognition.face_locations(resized_rgb_frame)
    unknown_face_encodings = face_recognition.face_encodings(resized_rgb_frame,unknown_face_locations)
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(dataset['encodings'],face_encoding)
        name = "Unknown"
        if True in matches:
            index_of_matched_label = matches.index(True)
            name =dataset['names'][index_of_matched_label]
        unknown_face_names.append(name)

    for (top,right,bottom,left),name in zip(unknown_face_locations,unknown_face_names):
        top *= 4
        right *= 4
        bottom *=4
        left *=4
        cv2.rectangle(frame, (left, top), (right, bottom), Colors.primary.value, 1)
        cv2.rectangle(frame,(left,top-20),(right,top),Colors.primary.value,cv2.FILLED)
        cv2.putText(frame, name, (left + 6, top - 6), font,1 , Colors.white.value, 1)
    
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
  
    fps = int(fps) 
  
    fps = str(fps) 

    database_name = "version_1.pickle"

    cv2.putText(frame, "FPS: "+ fps, (7,30), font, 1.5 , Colors.secondary.value,3)
    cv2.putText(frame, "Source: "+ database_name, (7,60), font, 1.5 , Colors.secondary.value,3)
    cv2.imshow("Cam",frame)


    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



