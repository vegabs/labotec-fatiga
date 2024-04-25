import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import deque
import joblib
import threading
import pygame
import argparse
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize

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
width = 640
height = 480

frame_rate = 23
input_rate = 5
count_rate = 0
prev = 0

last_category = ''
period_analyzes = 40
total_long_blink = 0
total_short_blink = 0
total = 0
consecutive_long = 0

ear_vector = deque(maxlen=15)
prediction_vector = deque(maxlen=5)

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# mediapipe settings ---------
mp_facemesh = mp.solutions.face_mesh.FaceMesh(False, 1, True, 0.5)
mp_drawing = mp.solutions.drawing_utils
mp_circle = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(255,0,0))
mp_line = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# font settings -------- 
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.8
font_color = (0,255,255)
font_tich = 2

# eye settings ----------
# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs


# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
fps_avg_frame_count = 10

detection_result_list = []

def visualize_callback(result: vision.ObjectDetectorResult,
                        output_image: mp.Image, timestamp_ms: int):
    result.timestamp_ms = timestamp_ms
    detection_result_list.append(result)

# Initialize the object detection model
base_options = python.BaseOptions(model_asset_path='model.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                        running_mode=vision.RunningMode.LIVE_STREAM,
                                        #display_names='es',
                                        #category_allowlist=['cell phone'],
                                        score_threshold=0.4,
                                        result_callback=visualize_callback)
detector = vision.ObjectDetector.create_from_options(options)

while True:
    time_elapsed = time.time() - prev
    ignore, frame = cam.read()
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    frame.flags.writeable = False

    results = mp_facemesh.process(frame)

    frame.flags.writeable = True

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    counter += 1

    img_h, img_w, img_c = frame.shape

    if time_elapsed > 1./frame_rate:
        prev = time.time()

        faces_found = results.multi_face_landmarks

        face_3d = []
        face_2d = []

        if faces_found != None:
            for this_face_landmark in faces_found:

                # OBJECT DETECTION
                # Run object detection using the model.
                detector.detect_async(mp_image, counter)
                current_frame = mp_image.numpy_view()
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

                # Calculate the FPS
                if counter % fps_avg_frame_count == 0:
                    end_time = time.time()
                    fps = fps_avg_frame_count / (end_time - start_time)
                    start_time = time.time()

                # Show the FPS
                fps_text = 'FPS = {:.1f}'.format(fps)
                text_location = (left_margin, row_size)
                # cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            #font_size, font_color, font_tich)

                # HEAD POSITION
                for idx, lm in enumerate(this_face_landmark.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
            

                # See where the user's head tilting
                if y < -10:
                    text = "Mirando Izquierda"
                elif y > 10:
                    text = "Mirando Derecha"
                elif x < -10:
                    text = "Mirando Abajo"
                elif x > 10:
                    text = "Mirando Arriba"
                else:
                    text = "Enfrente"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                
                text_s = 25
                cv2.putText(frame, text, (1, text_s), font, font_size, font_color, font_tich, cv2.LINE_AA)
                cv2.putText(frame, "x, pitch: " + str(np.round(x,2)), (1, text_s * 2), font, font_size, font_color, font_tich, cv2.LINE_AA)
                cv2.putText(frame, "y, yaw: " + str(np.round(y,2)), (1,  text_s * 3), font, font_size, font_color, font_tich, cv2.LINE_AA)
                cv2.putText(frame, "z, roll: " + str(np.round(z,2)), (1,  text_s * 4), font, font_size, font_color, font_tich, cv2.LINE_AA)
                cv2.line(frame, p1, p2, (255, 0, 0), 3)
                
                # calculate EAR
                EAR, _ = calculate_avg_ear(this_face_landmark.landmark, chosen_left_eye_idxs, chosen_right_eye_idxs, width, height)
                ear_vector.append(EAR)
                count_rate = count_rate + 1
                
                if count_rate == 5:
                    if len(ear_vector) > 14:
                        input_vector = ear_vector
                        predicted_category = predict_category(input_vector)
                        last_category = predicted_category
                        prediction_vector.append(predicted_category)

                    # check 1st condition: if the last 5 predictions where long blink
                    if len(prediction_vector) > 3:
                        all_long_blink = all(element == 'long_blink' for element in prediction_vector)
                        if all_long_blink:
                            sound_thread = threading.Thread(target=play_sound)
                            sound_thread.start() # play alarm

                    # check 2nd condition:
                    # if True:
                    #     prop = total_long_blink / (total_long_blink + total_short_blink)
                    #     if prop >= 0.25:
                    #         sound_thread = threading.Thread(target=play_sound)
                    #         sound_thread.start() # play alarm

                    count_rate = 0
                
                cv2.putText(frame, f"Estado ojos: {last_category}", (1,  text_s * 5), font, font_size, font_color, font_tich, cv2.LINE_AA)
                cv2.putText(frame, f"EAR: {round(EAR, 2)}", (1,  text_s * 6), font, font_size, font_color, font_tich, cv2.LINE_AA) 
 
                print(f'[INFO] Ojos: {last_category}, Celular: 0, Cigarro: 0, Distraccion: 0')  

                mp_drawing.draw_landmarks(frame, this_face_landmark, mp.solutions.face_mesh.FACEMESH_CONTOURS, mp_circle, mp_line)
                
    
    if detection_result_list:
        print(detection_result_list)
        vis_image = visualize(current_frame, detection_result_list[0])
        cv2.imshow('object_detector', vis_image)
        detection_result_list.clear()
    else:
        cv2.imshow('object_detector', frame)

    # cv2.imshow("Detector de fatiga", frame)
    cv2.moveWindow("Detector de fatiga", 1, 1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
