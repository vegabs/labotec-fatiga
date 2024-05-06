import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import joblib
import threading
import pygame
import argparse
import sys
import json
from collections import deque
from mediapipe.tasks import python
import simpleaudio as sa
from mediapipe.tasks.python import vision
from utils import visualize

# ----------------------------------------------------------------------------


def load_variables_from_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    for key, value in data.items():
        globals()[key] = value


def show_text(my_frame, my_text, id):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_color = (0, 0, 0)
    font_tich = 1
    text_pos_y_base = 20
    text_pos_x_base = 480
    cv2.putText(
        my_frame,
        my_text,
        (text_pos_x_base, text_pos_y_base * id),
        font,
        font_size,
        font_color,
        font_tich,
        cv2.LINE_AA,
    )


def play_sound():
    #pygame.mixer.init()
    #sound = pygame.mixer.Sound(AUDIO_FILENAME)
    #sound.play()
    # pygame.time.wait(int(sound.get_length() * 50))
    # pygame.mixer.quit()
    wave_obj = sa.WaveObject.from_wave_file(AUDIO_FILENAME)
    play_obj = wave_obj.play()


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
        frame_width: (int) WIDTH of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
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

    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(
        landmarks, right_eye_idxs, image_w, image_h
    )
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def visualize_callback(
    result: vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms: int
):
    result.timestamp_ms = timestamp_ms
    detection_result_list.append(result)


def get_head_position(face_landmark, face_2d, face_3d, img_h, img_w):
    for idx, lm in enumerate(face_landmark):
        if (
            idx == 33
            or idx == 263
            or idx == 1
            or idx == 61
            or idx == 291
            or idx == 199
        ):
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y]) # Get the 2D Coordinates
            face_3d.append([x, y, lm.z]) # Get the 3D Coordinates

    face_2d = np.array(face_2d, dtype=np.float64) # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64) # Convert it to the NumPy array

    focal_length = 1 * img_w # The camera matrix

    cam_matrix = np.array(
        [
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1],
        ]
    )

    dist_matrix = np.zeros((4, 1), dtype=np.float64) # The distortion parameters
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix) # Solve PnP
    rmat, jac = cv2.Rodrigues(rot_vec) # Get rotational matrix
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat) # Get angles

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Display the nose direction
    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    
    return face_2d, face_3d, p1, p2, x, y, z

# ----------------------------------------------------------------------------

# GENERAL SETTINGS
json_file = "settings.json"
load_variables_from_json(json_file)

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

# OPENCV SETTINGS
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# EAR SETTINGS
# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
ear_vector = deque(maxlen=15)
prediction_vector = deque(maxlen=5)
last_category = ""
period_analyzes = 40
total_long_blink = 0
total_short_blink = 0
total = 0
consecutive_long = 0
svm_model = joblib.load(MODEL_FILENAME)

# DISTRACTION PARAMETERS
distraction_left = deque(maxlen=5)
distraction_right = deque(maxlen=5)
distraction_top = deque(maxlen=5)
distraction_bottom = deque(maxlen=5)
status_head = 0

# MEDIAPIPE SETTINGS
mp_facemesh = mp.solutions.face_mesh.FaceMesh(False, 1, True, 0.5)
mp_drawing = mp.solutions.drawing_utils
mp_circle = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 255))
mp_line = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 255))
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates
cellphone_detection = deque(maxlen=10)

# Visualization parameters
fps_avg_frame_count = 10
frame_rate = 23
input_rate = 5
count_rate = 0
prev = 0

# AUDIO ALERT
alert_raised = False

# OBJECT DETECTION SETTINGS
status_cellphone = "NO"
status_cigarrete = "NO"
detection_result_list = []
base_options = python.BaseOptions(model_asset_path=OBJECT_DETECTION_MODEL_FILENAME)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    # display_names='es',
    # category_allowlist=['cell phone'],
    score_threshold=0.7,
    result_callback=visualize_callback,
)
detector = vision.ObjectDetector.create_from_options(options)

while True:
    time_elapsed = time.time() - prev

    ignore, frame = cam.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    frame.flags.writeable = False

    results = mp_facemesh.process(frame)

    frame.flags.writeable = True

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    counter += 1

    img_h, img_w, img_c = frame.shape

    alert_raised = False

    if time_elapsed > 1.0 / frame_rate:
        prev = time.time()

        faces_found = results.multi_face_landmarks

        face_3d = []
        face_2d = []

        if faces_found != None:
            for this_face_landmark in faces_found:

                # OBJECT DETECTION ------------------------------------------------------------------------------------
                # -----------------------------------------------------------------------------------------------------
                detector.detect_async(mp_image, counter)
                current_frame = mp_image.numpy_view()
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
                
                # print(cellphone_detection)
                # print("-----------")
                if cellphone_detection.count(1) >= 3:
                    alert_raised = True

                # # Calculate the FPS
                # if counter % fps_avg_frame_count == 0:
                #     end_time = time.time()
                #     fps = fps_avg_frame_count / (end_time - start_time)
                #     start_time = time.time()

                # # Show the FPS
                # fps_text = "FPS = {:.1f}".format(fps)

                # HEAD POSITION & DISTRACTION -------------------------------------------------------------------
                # -----------------------------------------------------------------------------------------------
                face_2d, face_3d, p1, p2, x, y, z = get_head_position(this_face_landmark.landmark, face_2d, face_3d, img_h, img_w)

                # See where the user's head tilting
                if y < -15:
                    text = "Mirando Izquierda"
                    status_head = -1
                    distraction_left.append(1)
                    distraction_right.append(0)
                    distraction_bottom.append(0)
                    distraction_top.append(0)
                elif y > 15:
                    text = "Mirando Derecha"
                    status_head = 1
                    distraction_left.append(0)
                    distraction_right.append(1)
                    distraction_bottom.append(0)
                    distraction_top.append(0)
                elif x < -15:
                    text = "Mirando Abajo"
                    status_head = -2
                    distraction_left.append(0)
                    distraction_right.append(0)
                    distraction_bottom.append(1)
                    distraction_top.append(0)
                elif x > 15:
                    text = "Mirando Arriba"
                    status_head = 2
                    distraction_left.append(0)
                    distraction_right.append(0)
                    distraction_bottom.append(0)
                    distraction_top.append(1)
                else:
                    text = "Enfrente"
                    status_head = 0
                    distraction_left.append(0)
                    distraction_right.append(0)
                    distraction_bottom.append(0)
                    distraction_top.append(0)
                
                if distraction_top.count(1) >= 3 or distraction_bottom.count(1) >= 3 or distraction_left.count(1) >= 3 or distraction_right.count(1) >= 3:
                    alert_raised = True

                cv2.line(frame, p1, p2, (0, 0, 0), 2)

                # EAR & DROWNSINESS -------------------------------------------------------------------
                # -----------------------------------------------------------------------------------------------
                EAR, _ = calculate_avg_ear(
                    this_face_landmark.landmark,
                    chosen_left_eye_idxs,
                    chosen_right_eye_idxs,
                    WIDTH,
                    HEIGHT,
                )
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
                        all_long_blink = all(
                            element == "long_blink" for element in prediction_vector
                        )
                        if all_long_blink:
                            alert_raised = True

                    # check 2nd condition:
                    # if True:
                    #     prop = total_long_blink / (total_long_blink + total_short_blink)
                    #     if prop >= 0.25:
                    #         sound_thread = threading.Thread(target=play_sound)
                    #         sound_thread.start() # play alarm

                    count_rate = 0

				# USER INTERFACE --------------------------------------------------------------------------------
				# -----------------------------------------------------------------------------------------------
                
				# DRAW FACE LANDMARKS
                mp_drawing.draw_landmarks(
                    frame,
                    this_face_landmark,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    mp_circle,
                    mp_line,
                )
                
                # DISPLAY TEXTS
                show_text(frame, text, 1)
                show_text(frame, "x, pitch: " + str(np.round(x, 2)), 2)
                show_text(frame, "y, yaw: " + str(np.round(y, 2)), 3)
                show_text(frame, "z, roll: " + str(np.round(z, 2)), 4)
                show_text(frame, f"Estado ojos: {last_category}", 5)
                show_text(frame, f"EAR: {round(EAR, 2)}", 6)
                show_text(frame, f"Celular: {status_cellphone}", 7)
                show_text(frame, f"Cigarro: {status_cigarrete}", 8)

                # SERIAL COMMUNICATION --------------------------------------------------------------------------------
                # print(f'[INFO]')

                # AUDIO THINGS
                if alert_raised:
                    print('audio XXXXXXX')
                    sound_thread = threading.Thread(target=play_sound)
                    sound_thread.start() # play alarm

    if detection_result_list:
        # print(detection_result_list)
        vis_image = visualize(current_frame, detection_result_list[0])

        # DETECTION RESULTS
        for detection in detection_result_list[0].detections:
            category = detection.categories[0]
            category_name = category.category_name
            if category_name == "cellphone":
                status_cellphone = "SI"
                cellphone_detection.append(1)
            else:
                status_cellphone = "NO"
                cellphone_detection.append(0)

            if category_name == "cigarette":
                status_cigarrete = "SI"
            else:
                status_cigarrete = "NO"

        cv2.imshow("object_detector", vis_image)
        detection_result_list.clear()
    else:
        cellphone_detection.append(0)
        status_cellphone = "NO"
        status_cigarrete = "NO"
        cv2.imshow("object_detector", frame)

    # cv2.imshow("Detector de fatiga", frame)
    cv2.moveWindow("Detector de fatiga", 1, 1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
