import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import deque
import joblib
import threading
import pygame

print(cv2.__version__)

# FILES
MODEL_FILENAME = 'models/svm_modelv2.pkl'
AUDIO_FILENAME = 'media/noti.wav'

# Load the exported SVM model
svm_model = joblib.load(MODEL_FILENAME)

def play_sound():
    pygame.mixer.init()
    sound = pygame.mixer.Sound(AUDIO_FILENAME)
    sound.play()
    pygame.time.wait(int(sound.get_length() * 1000))
    pygame.mixer.quit()

def predict_category(input_vector):
    # Convert input_vector to a numpy array if it's not already
    input_vector = np.array(input_vector).reshape(1, -1)  # Reshape to 2D array
    
    # Make predictions using the loaded model
    predicted_category = svm_model.predict(input_vector)
    
    return predicted_category[0]

def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.
 
    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame
 
    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, 
                                             frame_width, frame_height)
            coords_points.append(coord)
 
        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])
 
        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
 
    except:
        ear = 0.0
        coords_points = None
 
    return ear, coords_points
	
def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """Calculate Eye aspect ratio"""
 
    left_ear, left_lm_coordinates = get_ear(
                                      landmarks, 
                                      left_eye_idxs, 
                                      image_w, 
                                      image_h
                                    )
    right_ear, right_lm_coordinates = get_ear(
                                      landmarks, 
                                      right_eye_idxs, 
                                      image_w, 
                                      image_h
                                    )
    Avg_EAR = (left_ear + right_ear) / 2.0
 
    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

# CV2 settings ---------
width = 1280
height = 720

frame_rate = 23
input_rate = 5
count_rate = 0
prev = 0

last_category = ''
period_analyzes = 60
total_long_blink = 0
total_short_blink = 0
total = 0
consecutive_long = 0

ear_vector = deque(maxlen=15)
prediction_vector = deque(maxlen=5)

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

                    if len(ear_vector) > 14:
                        input_vector = ear_vector # Example input vector with 15 elements
                        predicted_category = predict_category(input_vector)
                        last_category = predicted_category
                        #print(f"Predicted Category: {predicted_category}")

                        prediction_vector.append(predicted_category)
                    
                    if len(prediction_vector) > 4:
                        print('estoy aqui')
                        all_long_blink = all(element == 'long_blink' for element in prediction_vector)
                        if all_long_blink:
                            print('DORMIDOOOOOO!')
                            sound_thread = threading.Thread(target=play_sound)

                            # Start the thread
                            sound_thread.start()

                    count_rate = 0
                
                cv2.putText(frame, f"CATEGORY: {last_category}", (1, 60), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"EAR: {round(EAR, 2)}", (1, 24), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)    
    
    cv2.imshow("Detector de fatiga", frame)
    cv2.moveWindow("Detector de fatiga", 1, 1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
