import cv2
import time
import mediapipe as mp
import numpy as np
import time
import argparse
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
from PIL import Image

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

model = 'efficientdet_lite0.tflite'
num_threads = 4

dispW = 640
dispH = 480

HEAD_DETECTION_RANGE = 10
x = 0
y = 0
z = 0
face_2d = []
face_3d = []

cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cam.set(cv2.CAP_PROP_FPS, 30)

pos = (20, 60)
font = cv2.FONT_HERSHEY_SIMPLEX
height = 1.5
weight = 3
myColor = (255, 0, 0)

fps = 0

# OBJECT DETECTION VARIABLES
status_cellphone = False
GLASSES_STATE = False

# EAR SETTINGS
# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
last_category = ""

# DISTRACTION PARAMETERS
status_head = 0

# MEDIAPIPE SETTINGS
mp_facemesh = mp.solutions.face_mesh.FaceMesh(False, 1, True, 0.5)
mp_drawing = mp.solutions.drawing_utils
mp_circle = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(255, 255, 255))
mp_line = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(255, 255, 255))
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# OBJECT DETECTION SETTINGS
base_options = core.BaseOptions(
    file_name=model, use_coral=False, num_threads=num_threads)
detection_options = processor.DetectionOptions(
    max_results=3, score_threshold=.5, category_name_allowlist=["cell phone"])
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)
tStart = time.time()


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
            coord = denormalize_coordinates(
                lm.x, lm.y, frame_width, frame_height)
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
            face_2d.append([x, y])  # Get the 2D Coordinates
            face_3d.append([x, y, lm.z])  # Get the 3D Coordinates

    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)
    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w  # The camera matrix

    cam_matrix = np.array(
        [
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1],
        ]
    )

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix)  # Solve PnP
    rmat, jac = cv2.Rodrigues(rot_vec)  # Get rotational matrix
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # Get angles

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Display the nose direction
    nose_3d_projection, jacobian = cv2.projectPoints(
        nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

    return face_2d, face_3d, p1, p2, x, y, z


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """Calculate Eye aspect ratio"""
    left_ear, left_lm_coordinates = get_ear(
        landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(
        landmarks, right_eye_idxs, image_w, image_h
    )
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def get_glasses(img_org, landmarks, frame_width, frame_height):
    # GLASSES SETTINGS
    glass_id = [55, 285, 174, 399]
    coords_points = []

    for i in glass_id:
        lm = landmarks[i]
        coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
        coords_points.append(coord)

    x_values = [coord[0] for coord in coords_points]
    y_values = [coord[1] for coord in coords_points]

    # Find the maximum and minimum values for x and y
    max_x = max(x_values)
    min_x = min(x_values)
    max_y = max(y_values)
    min_y = min(y_values)

    img_org = Image.fromarray(img_org.astype('uint8'), 'RGB')
    img2 = img_org.crop((min_x, min_y, max_x, max_y))
    img_blur = cv2.GaussianBlur(np.array(img2), (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    edges_center = edges.T[(int(len(edges.T)/2))]
    if 255 in edges_center:
        return True
    else:
        return False

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

# MEDIAPIPE SETTINGS
mp_facemesh = mp.solutions.face_mesh.FaceMesh(False, 1, True, 0.5)
mp_drawing = mp.solutions.drawing_utils
mp_circle = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(255, 255, 255))
mp_line = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(255, 255, 255))
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

while True:
    ret, im = cam.read()
    im = cv2.resize(im, (dispW, dispH))
    im = cv2.flip(im, 1)
    imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imTensor = vision.TensorImage.create_from_array(imRGB)
    detections = detector.detect(imTensor)
    image = utils.visualize(im, detections)

    results = mp_facemesh.process(im)
    faces_found = results.multi_face_landmarks

    if faces_found != None:
        for this_face_landmark in faces_found:

            # GLASSES
            GLASSES_STATE = get_glasses(
                im, this_face_landmark.landmark, dispW, dispH)

            # HEAD POSITION & DISTRACTION -------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------
            face_2d, face_3d, p1, p2, x, y, z = get_head_position(
                this_face_landmark.landmark, face_2d, face_3d, dispH, dispH)

            # See where the user's head tilting
            if y < HEAD_DETECTION_RANGE * -1:
                text = "Mirando Izquierda"
                status_head = -1
            elif y > HEAD_DETECTION_RANGE:
                text = "Mirando Derecha"
                status_head = 1
            elif x < HEAD_DETECTION_RANGE * -1:
                text = "Mirando Abajo"
                status_head = -2
            elif x > HEAD_DETECTION_RANGE:
                text = "Mirando Arriba"
                status_head = 2
            else:
                text = "Enfrente"
                status_head = 0

            cv2.line(im, p1, p2, (0, 0, 0), 2)

            # EAR & DROWNSINESS -------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------
            EAR, _ = calculate_avg_ear(
                this_face_landmark.landmark,
                chosen_left_eye_idxs,
                chosen_right_eye_idxs,
                dispW,
                dispH,
            )

    # OBJECT DETECTIONS
    if len(detections.detections) > 0:
        for i in detections.detections:
            if i.categories[0].category_name == 'cell phone':
                status_cellphone = True
    else:
        status_cellphone = False

    show_text(im, str(int(fps))+' FPS', 1)
    show_text(im,  f"Celular: {status_cellphone}", 2)
    show_text(im, f"Lentes: {str(GLASSES_STATE)}", 3)
    show_text(im, "x, pitch: " + str(np.round(x, 2)), 4)
    show_text(im, "y, yaw: " + str(np.round(y, 2)), 5)
    show_text(im, "z, roll: " + str(np.round(z, 2)), 6)
    show_text(im, f"Estado ojos: {last_category}", 7)
    show_text(im, f"EAR: {round(EAR, 2)}", 8)

    cv2.imshow('Detector Fatiga', im)

    if cv2.waitKey(1) == ord('q'):
        break
    tEnd = time.time()
    loopTime = tEnd-tStart
    fps = .9*fps + .1*1/loopTime
    tStart = time.time()
cv2.destroyAllWindows()
