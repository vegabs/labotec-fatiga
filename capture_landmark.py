import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import deque
from calculations import *

print(cv2.__version__)

ear_vector = deque(maxlen=15)

frame_rate = 23
input_rate = 5
count_rate = 0
prev = 0

csv_file = '__gaby-openv4.csv'

# CV2 settings ---------
width = 1280
height = 720

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# mediapipe settings ---------
mp_facemesh = mp.solutions.face_mesh.FaceMesh(False, 1, True, 0.5)
mp_drawing = mp.solutions.drawing_utils
mp_circle = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(255,0,0))
mp_line = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0,255,0))
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# font settings -------- 
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_color = (0,255,255)
font_tich = 1

# eye settings ----------
# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)

    while True:
        time_elapsed = time.time() - prev
        ignore, frame = cam.read()

        if time_elapsed > 1./frame_rate:
            prev = time.time()

            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            results = mp_facemesh.process(frame_rgb)
            faces_found = results.multi_face_landmarks
            # print(results.multi_face_landmarks)

            if faces_found != None:
                for this_face_landmark in faces_found:
                    mp_drawing.draw_landmarks(frame, this_face_landmark, mp.solutions.face_mesh.FACEMESH_CONTOURS, mp_circle, mp_line)

                    # indx = 0

                    # for lm in this_face_landmark.landmark:
                    #     if indx in all_chosen_idxs:
                    #         esta = True
                    #     else:
                    #         esta = False

                    #     if esta:
                    #         cv2.putText(frame, str(indx), (int(lm.x * width), int(lm.y * height)), font, font_size, font_color, font_tich)
                    #     print(indx)
                    #     indx = indx + 1


                    EAR, _ = calculate_avg_ear(this_face_landmark.landmark, chosen_left_eye_idxs, chosen_right_eye_idxs, width, height)
                    count_rate = count_rate + 1
                    ear_vector.append(EAR)
                    
                    if count_rate == 5:
                        #print(len(ear_vector))
                        #print(ear_vector)
                        #print(count_rate)
                        writer.writerow(ear_vector)
                        count_rate = 0

                    
                    #print(count_rate)

                    cv2.putText(frame, f"EAR: {round(EAR, 2)}", (1, 24), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)    

        
        cv2.imshow("my WEBcam", frame)
        cv2.moveWindow("my WEBcam", 1, 1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
