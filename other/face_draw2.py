import cv2
import mediapipe as mp

print(cv2.__version__)
width = 1280
height = 720
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

faceMesh = mp.solutions.face_mesh.FaceMesh(False, 1, True, 0.5)
mpDraw = mp.solutions.drawing_utils

drawSpecCircle = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(255,0,0))
drawSpecLine = mpDraw.DrawingSpec(thickness=3, circle_radius=1, color=(0,255,0))

font = cv2.FONT_HERSHEY_SIMPLEX
fontSize = 0.5
fontColor = (0,255,255)
fontThick = 1

while True:
    ignore, frame = cam.read()
    frame = cv2.resize(frame, (width, height))
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)
    print(results.multi_face_landmarks)

    if results.multi_face_landmarks != None:
        for faceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLandmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS, drawSpecCircle, drawSpecLine)

            indx = 0
            for lm in faceLandmarks.landmark:
                cv2.putText(frame, str(indx), (int(lm.x * width), int(lm.y * height)), font, fontSize, fontColor, fontThick)
                indx = indx + 1

    cv2.imshow("my WEBcam", frame)
    cv2.moveWindow("my WEBcam", 1, 1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
